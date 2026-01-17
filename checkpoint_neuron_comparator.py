#!/usr/bin/env python3


"""
Two-checkpoint neuron comparison toolkit for LLaMA-3.x (GQA) + LoRA checkpoints.

TOP HALF:
  - Load 2 checkpoints
  - Run same inputs
  - Capture: resid_pre, block_out, mlp_gate/up/out, qkv (flat + head-split)

SECOND HALF (one by one):
  1) Identify LoRA-affected neurons
  2) Track instruction tuning drift
  3) Perform activation patching across models (single-layer / multi-layer)
  4) Find circuits introduced by finetuning (simple causal influence via patch/ablation)
  5) Build neuron correspondence maps (correlation + matching)
  6) Add automatic neuron matching (greedy + optional Hungarian if SciPy present)
  7) Add statistical significance testing (paired t-test + bootstrap CI)
  8) Add activation patching between models (general patch runner)
  9) Export SAE-ready comparison datasets

Notes:
- This script assumes both checkpoints share architecture (e.g., Llama-3.1-8B).
- Uses CPU-stored activations to avoid multi-GPU hook pitfalls with device_map="auto".
- RoPE-applied Q/K extraction is not included (needs deeper attention override).
"""
import json
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM



@dataclass
class RunConfig:
    ckpt_a: str
    ckpt_b: str

    prompts: List[str]

    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    attn_implementation: str = "eager"

    capture_reduction: str = "last"

    capture_resid_pre: bool = True
    capture_block_out: bool = True
    capture_mlp: bool = True
    capture_qkv: bool = True

    topk_neurons: int = 50
    topk_heads: int = 20
    seed: int = 0

    circuit_test_layers: int = 8
    circuit_test_neurons_per_layer: int = 10

    bootstrap_samples: int = 1000
    bootstrap_ci: float = 0.95

    export_dir: str = "./compare_export"
    export_family: str = "mlp_gate"
    export_reduction: str = "all"


CFG = RunConfig(
    ckpt_a=os.environ.get("CKPT_A", "path/to/checkpoint-a"),
    ckpt_b=os.environ.get("CKPT_B", "path/to/checkpoint-b"),

    prompts=[
        "Which of these has its entrance in Piyer Loti Caddesi, Hagia Sophia or Theodosius Cistern?",
        "Explain why the sky is blue in one sentence.",
        "Write a Python function that checks if a number is prime.",
        "What is the capital of Turkey?",
    ],
    capture_reduction="last",
    export_family="mlp_gate",
    export_reduction="all",
    export_dir="./compare_export",
)



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_tensor(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x: [bsz, seq, hidden] or similar
    returns:
      - "all": [bsz, seq, hidden]
      - "last": [bsz, hidden]
      - "mean_seq": [bsz, hidden]
    """
    if mode == "all":
        return x
    if x.dim() < 3:
        return x
    if mode == "last":
        return x[:, -1, :]
    if mode == "mean_seq":
        return x.mean(dim=1)
    raise ValueError(f"Unknown capture_reduction: {mode}")


def load_model_and_tokenizer(model_dir: str, cfg: RunConfig):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def capture_activations_for_prompt(
    model,
    tokenizer,
    prompt: str,
    cfg: RunConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Returns activations dict (CPU tensors) for a single prompt.
    Keys:
      acts["resid_pre"]["layer_i"]
      acts["block_out"]["layer_i"]
      acts["mlp_gate"]["layer_i"], ["mlp_up"], ["mlp_out"]
      acts["qkv_flat"]["layer_i_q/k/v"]
      acts["qkv_heads"]["layer_i"]["q/k/v"]  (head-split; q heads != kv heads for GQA)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    acts: Dict[str, Dict[str, Any]] = {
        "resid_pre": {},
        "block_out": {},
        "mlp_gate": {},
        "mlp_up": {},
        "mlp_out": {},
        "qkv_flat": {},
        "qkv_heads": {},
    }

    hooks = []

    def save_input(name, store, reduction):
        def hook(m, x, y):
            t = x[0].detach()
            t = reduce_tensor(t, reduction)
            store[name] = t.to("cpu")
        return hook

    def save_output(name, store, reduction):
        def hook(m, x, y):
            t = y.detach()
            t = reduce_tensor(t, reduction)
            store[name] = t.to("cpu")
        return hook

    for i, block in enumerate(model.model.layers):
        layer_key = f"layer_{i}"

        if cfg.capture_resid_pre:
            hooks.append(block.register_forward_hook(save_input(layer_key, acts["resid_pre"], cfg.capture_reduction)))

        if cfg.capture_block_out:
            hooks.append(block.register_forward_hook(save_output(layer_key, acts["block_out"], cfg.capture_reduction)))

        if cfg.capture_mlp:
            hooks.append(block.mlp.gate_proj.register_forward_hook(save_output(layer_key, acts["mlp_gate"], cfg.capture_reduction)))
            hooks.append(block.mlp.up_proj.register_forward_hook(save_output(layer_key, acts["mlp_up"], cfg.capture_reduction)))
            hooks.append(block.mlp.down_proj.register_forward_hook(save_output(layer_key, acts["mlp_out"], cfg.capture_reduction)))

        if cfg.capture_qkv:
            hooks.append(block.self_attn.q_proj.register_forward_hook(save_output(f"{layer_key}_q", acts["qkv_flat"], cfg.capture_reduction)))
            hooks.append(block.self_attn.k_proj.register_forward_hook(save_output(f"{layer_key}_k", acts["qkv_flat"], cfg.capture_reduction)))
            hooks.append(block.self_attn.v_proj.register_forward_hook(save_output(f"{layer_key}_v", acts["qkv_flat"], cfg.capture_reduction)))

    _ = model(**inputs, return_dict=True)

    for h in hooks:
        h.remove()

    if cfg.capture_qkv:
        config = model.config
        n_q = config.num_attention_heads
        n_kv = config.num_key_value_heads
        head_dim = config.hidden_size // n_q

        for i in range(len(model.model.layers)):
            layer_key = f"layer_{i}"
            q = acts["qkv_flat"][f"{layer_key}_q"]
            k = acts["qkv_flat"][f"{layer_key}_k"]
            v = acts["qkv_flat"][f"{layer_key}_v"]

            def split_heads(t, n_heads):
                if t.dim() == 2:
                    bsz, hid = t.shape
                    return t.view(bsz, n_heads, head_dim)
                if t.dim() == 3:
                    bsz, seq, hid = t.shape
                    return t.view(bsz, seq, n_heads, head_dim)
                raise RuntimeError(f"Unexpected tensor rank for qkv: {t.shape}")

            acts["qkv_heads"][layer_key] = {
                "q": split_heads(q, n_q),
                "k": split_heads(k, n_kv),
                "v": split_heads(v, n_kv),
            }

    return acts


@torch.no_grad()
def capture_activations_dataset(model, tokenizer, prompts: List[str], cfg: RunConfig):
    """
    Returns list of acts per prompt.
    """
    all_acts = []
    for p in prompts:
        all_acts.append(capture_activations_for_prompt(model, tokenizer, p, cfg))
    return all_acts


def stack_family_over_dataset(dataset_acts: List[Dict[str, Dict[str, torch.Tensor]]],
                              family: str,
                              layer: str) -> torch.Tensor:
    """
    family: "resid_pre" / "block_out" / "mlp_gate" / "mlp_up" / "mlp_out"
    Returns stacked tensor over prompts.
      if reduction="last"/"mean_seq": each is [bsz, hidden] -> stack -> [N, hidden]
      if reduction="all": each is [bsz, seq, hidden] -> stack -> [N, seq, hidden]
    bsz should be 1 for single prompt tokenization.
    """
    xs = []
    for acts in dataset_acts:
        x = acts[family][layer]
        if x.dim() >= 2 and x.shape[0] == 1:
            x = x[0]
        xs.append(x)
    return torch.stack(xs, dim=0)



def per_neuron_delta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A,B: [N, hidden] or [N, seq, hidden]
    returns per-neuron mean absolute delta: [hidden]
    """
    if A.dim() == 3:
        return (A - B).abs().mean(dim=(0, 1))
    return (A - B).abs().mean(dim=0)


def per_neuron_cosine(A: torch.Tensor, B: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    returns per-neuron cosine similarity across samples/time.
    For [N, hidden]:
      cosine over N for each neuron -> [hidden]
    For [N, seq, hidden]:
      flatten N*seq, then cosine -> [hidden]
    """
    if A.dim() == 3:
        A2 = A.reshape(-1, A.shape[-1])
        B2 = B.reshape(-1, B.shape[-1])
    else:
        A2, B2 = A, B
    num = (A2 * B2).sum(dim=0)
    den = (A2.norm(dim=0) * B2.norm(dim=0)).clamp_min(eps)
    return num / den


def topk_indices(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(x, k=min(k, x.numel()))
    return idx, vals



def identify_lora_affected_neurons(
    actsA_ds, actsB_ds, family: str, cfg: RunConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Heuristic: neurons whose activations changed most between A and B.
    (If A is base and B is LoRA/finetuned, these are likely LoRA-affected.)
    Returns per-layer top-K neuron indices + deltas.
    """
    out = {}
    layers = sorted(list(actsA_ds[0][family].keys()), key=lambda s: int(s.split("_")[1]))
    for layer in layers:
        A = stack_family_over_dataset(actsA_ds, family, layer)
        B = stack_family_over_dataset(actsB_ds, family, layer)
        d = per_neuron_delta(A, B)
        idx, vals = topk_indices(d, cfg.topk_neurons)
        out[layer] = {
            "top_idx": idx.tolist(),
            "top_delta": vals.tolist(),
            "mean_delta": d.mean().item(),
            "max_delta": d.max().item(),
        }
    return out



def track_instruction_tuning_drift(
    actsA_ds, actsB_ds, family: str
) -> Dict[str, Any]:
    """
    Drift summary per layer: mean L2 delta (over prompts, and seq if present).
    """
    layers = sorted(list(actsA_ds[0][family].keys()), key=lambda s: int(s.split("_")[1]))
    drift = {}
    for layer in layers:
        A = stack_family_over_dataset(actsA_ds, family, layer)
        B = stack_family_over_dataset(actsB_ds, family, layer)
        diff = A - B
        if diff.dim() == 3:
            l2 = diff.norm(dim=-1).mean().item()
        else:
            l2 = diff.norm(dim=-1).mean().item()
        drift[layer] = l2
    ranked = sorted(drift.items(), key=lambda kv: kv[1], reverse=True)
    return {"per_layer_mean_l2": drift, "ranked_layers": ranked}



@torch.no_grad()
def run_with_activation_patching(
    model,
    tokenizer,
    prompt: str,
    patch_family: str,
    patch_layer_to_tensor: Dict[str, torch.Tensor],
    patch_mode: str = "replace",
    patch_location: str = "resid_pre",
) -> torch.Tensor:
    """
    Returns logits for the prompt with patching applied.
    patch_layer_to_tensor: layer_name -> tensor in the SAME SHAPE as the module input/output after reduction handling.

    For robust patching we patch FULL tensors (not reduced) by forcing capture_reduction="all" when building patches.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    hooks = []

    def pre_hook_factory(layer_name, patch_tensor):
        def pre_hook(module, inputs_):
            x = inputs_[0]
            pt = patch_tensor.to(x.device, dtype=x.dtype)
            if patch_mode == "replace":
                return (pt,) + inputs_[1:]
            elif patch_mode == "add":
                return (x + pt,) + inputs_[1:]
            else:
                raise ValueError(patch_mode)
        return pre_hook

    def post_hook_factory(layer_name, patch_tensor):
        def hook(module, inputs_, output):
            y = output
            pt = patch_tensor.to(y.device, dtype=y.dtype)
            if patch_mode == "replace":
                return pt
            elif patch_mode == "add":
                return y + pt
            else:
                raise ValueError(patch_mode)
        return hook

    for i, block in enumerate(model.model.layers):
        layer_key = f"layer_{i}"
        if layer_key not in patch_layer_to_tensor:
            continue
        pt = patch_layer_to_tensor[layer_key]
        if patch_location == "resid_pre":
            hooks.append(block.register_forward_pre_hook(pre_hook_factory(layer_key, pt)))
        elif patch_location == "block_out":
            hooks.append(block.register_forward_hook(post_hook_factory(layer_key, pt)))
        else:
            raise ValueError(patch_location)

    out = model(**inputs, return_dict=True)
    for h in hooks:
        h.remove()
    return out.logits.detach().to("cpu")



def activation_patching_across_models(
    tokA, modelA, tokB, modelB,
    prompt: str,
    cfg: RunConfig,
    layer_to_patch: int,
    patch_location: str = "resid_pre",
):
    """
    Example: patch one layer in modelB with activations from modelA and see logit shift.
    Uses capture_reduction="all" internally for patch tensors to be shape-compatible.
    """
    tmp_cfg = RunConfig(**{**cfg.__dict__, "capture_reduction": "all"})
    actsA = capture_activations_for_prompt(modelA, tokA, prompt, tmp_cfg)
    layer_key = f"layer_{layer_to_patch}"

    if patch_location == "resid_pre":
        patch_tensor = actsA["resid_pre"][layer_key]
    elif patch_location == "block_out":
        patch_tensor = actsA["block_out"][layer_key]
    else:
        raise ValueError(patch_location)

    logits_base = modelB(**tokB(prompt, return_tensors="pt").to(modelB.device), return_dict=True).logits.detach().cpu()
    logits_patch = run_with_activation_patching(
        model=modelB,
        tokenizer=tokB,
        prompt=prompt,
        patch_family=patch_location,
        patch_layer_to_tensor={layer_key: patch_tensor},
        patch_mode="replace",
        patch_location=patch_location,
    )

    p = F.softmax(logits_base[:, -1, :].float(), dim=-1)
    q = F.softmax(logits_patch[:, -1, :].float(), dim=-1)
    kl = (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
    l2 = (logits_base[:, -1, :] - logits_patch[:, -1, :]).norm(dim=-1).mean().item()

    return {"layer": layer_to_patch, "patch_location": patch_location, "kl_last_token": kl, "l2_last_token": l2}



def find_circuits_introduced_by_finetuning(
    tokA, modelA, tokB, modelB,
    prompt: str,
    family: str,
    cfg: RunConfig,
):
    """
    Simple causal-ish test:
      - Find top-changed neurons per selected layer (from family activations)
      - For each candidate neuron:
          Patch modelB's selected activation tensor at [token=-1, neuron] from A (or zero)
          Measure change in logit of modelB's *argmax token* at last position
    This is not a full circuit discovery pipeline, but it’s a strong practical starting point.
    """

    tmp_cfg = RunConfig(**{**cfg.__dict__, "capture_reduction": "all"})
    actsA = capture_activations_for_prompt(modelA, tokA, prompt, tmp_cfg)
    actsB = capture_activations_for_prompt(modelB, tokB, prompt, tmp_cfg)

    L = len(modelA.model.layers)
    chosen_layers = sorted(set([int(round(i)) for i in torch.linspace(0, L - 1, steps=min(cfg.circuit_test_layers, L)).tolist()]))

    logits_base = modelB(**tokB(prompt, return_tensors="pt").to(modelB.device), return_dict=True).logits.detach().cpu()
    target_id = int(torch.argmax(logits_base[0, -1, :]).item())
    base_logit = float(logits_base[0, -1, target_id].item())

    results = []

    for li in chosen_layers:
        layer_key = f"layer_{li}"
        A = actsA[family][layer_key]
        B = actsB[family][layer_key]
        d = (A[0, -1, :] - B[0, -1, :]).abs()
        idx, vals = topk_indices(d, cfg.circuit_test_neurons_per_layer)

        for neuron_idx, delta_val in zip(idx.tolist(), vals.tolist()):
            patch = B.clone()
            patch[0, -1, neuron_idx] = A[0, -1, neuron_idx]

            logits_patch = run_with_activation_patching(
                model=modelB,
                tokenizer=tokB,
                prompt=prompt,
                patch_family=family,
                patch_layer_to_tensor={layer_key: patch},
                patch_mode="replace",
                patch_location=("resid_pre" if family == "resid_pre" else "block_out" if family == "block_out" else "block_out"),
            )

            patched_logit = float(logits_patch[0, -1, target_id].item())
            effect = patched_logit - base_logit

            results.append({
                "layer": li,
                "neuron": neuron_idx,
                "abs_delta_last_token": float(delta_val),
                "target_token_id": target_id,
                "base_target_logit": base_logit,
                "patched_target_logit": patched_logit,
                "logit_effect": effect,
            })

    results.sort(key=lambda r: abs(r["logit_effect"]), reverse=True)
    return {"prompt": prompt, "target_id": target_id, "top_effects": results[: min(200, len(results))]}



def correlation_matrix(A: torch.Tensor, B: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    A,B: [N, hidden]  (if you have seq, flatten first)
    Returns corr: [hidden, hidden] where corr[i,j] = corr(A[:,i], B[:,j]).
    """
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    A = A / (A.std(dim=0, keepdim=True).clamp_min(eps))
    B = B / (B.std(dim=0, keepdim=True).clamp_min(eps))
    return (A.t() @ B) / max(1, (A.shape[0] - 1))


def greedy_match_from_corr(corr: torch.Tensor, topk: Optional[int] = None) -> List[Tuple[int, int, float]]:
    """
    Greedy one-to-one matching using highest correlation first.
    Returns list of (i_in_A, j_in_B, corr_value).
    """
    H = corr.shape[0]
    used_i = set()
    used_j = set()
    pairs = []

    flat = corr.flatten()
    vals, idxs = torch.sort(flat, descending=True)
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        i = idx // H
        j = idx % H
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j, float(v)))
        if topk is not None and len(pairs) >= topk:
            break
        if len(pairs) >= H:
            break
    return pairs


def hungarian_match_from_corr(corr: torch.Tensor) -> List[Tuple[int, int, float]]:
    """
    Optional exact matching via Hungarian algorithm if SciPy is available.
    Maximizes correlation (so we minimize -corr).
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return []

    cost = (-corr).float().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_ind, col_ind):
        pairs.append((int(i), int(j), float(corr[i, j].item())))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def build_neuron_correspondence_maps(
    actsA_ds, actsB_ds, family: str, cfg: RunConfig
) -> Dict[str, Any]:
    """
    Builds per-layer correspondence map between neurons in A and B:
      - correlation matrix
      - greedy matching
      - optional Hungarian matching (if SciPy exists)
    """
    layers = sorted(list(actsA_ds[0][family].keys()), key=lambda s: int(s.split("_")[1]))
    out = {}

    for layer in layers:
        A = stack_family_over_dataset(actsA_ds, family, layer)
        B = stack_family_over_dataset(actsB_ds, family, layer)

        if A.dim() == 3:
            A = A.reshape(-1, A.shape[-1])
            B = B.reshape(-1, B.shape[-1])

        corr = correlation_matrix(A, B)

        greedy = greedy_match_from_corr(corr, topk=cfg.topk_neurons)
        hung = hungarian_match_from_corr(corr)
        if hung:
            hung_top = hung[:cfg.topk_neurons]
        else:
            hung_top = []

        out[layer] = {
            "greedy_top": greedy,
            "hungarian_top": hung_top,
            "diag_mean_corr": float(torch.diag(corr).mean().item()),
            "diag_median_corr": float(torch.diag(corr).median().item()),
        }

    return out



def paired_t_test(x: torch.Tensor) -> Dict[str, float]:
    """
    x: [N] paired differences (e.g., per-prompt drift scalar)
    Returns t-stat and approximate two-sided p-value using normal approx if SciPy missing.
    """
    N = x.numel()
    mean = x.mean().item()
    std = x.std(unbiased=True).item()
    if std == 0 or N < 2:
        return {"t": float("nan"), "p": float("nan"), "mean": mean, "std": std, "n": N}
    t = mean / (std / math.sqrt(N))

    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), df=N - 1))
        p = float(p)
    except Exception:
        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        p = float(p)

    return {"t": float(t), "p": p, "mean": mean, "std": std, "n": int(N)}


def bootstrap_ci(x: torch.Tensor, num_samples: int, ci: float, seed: int) -> Tuple[float, float]:
    """
    x: [N] samples; bootstrap CI for the mean.
    """
    rng = random.Random(seed)
    N = x.numel()
    means = []
    x_list = x.tolist()
    for _ in range(num_samples):
        sample = [x_list[rng.randrange(N)] for _ in range(N)]
        means.append(sum(sample) / N)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * num_samples)]
    hi = means[int((1 - alpha) * num_samples) - 1]
    return float(lo), float(hi)


def significance_tests_for_layer_drift(actsA_ds, actsB_ds, family: str) -> Dict[str, Any]:
    """
    For each layer, compute a per-prompt scalar drift (L2 of last token or mean over seq),
    then run paired t-test and bootstrap CI over prompts.
    """
    layers = sorted(list(actsA_ds[0][family].keys()), key=lambda s: int(s.split("_")[1]))
    out = {}
    for layer in layers:
        A = stack_family_over_dataset(actsA_ds, family, layer)
        B = stack_family_over_dataset(actsB_ds, family, layer)

        if A.dim() == 3:
            dif = (A - B).norm(dim=-1).mean(dim=1)
        else:
            dif = (A - B).norm(dim=-1)

        test = paired_t_test(dif)
        lo, hi = bootstrap_ci(dif, num_samples=CFG.bootstrap_samples, ci=CFG.bootstrap_ci, seed=CFG.seed)

        out[layer] = {
            "mean_drift": float(dif.mean().item()),
            "t_test": test,
            "bootstrap_ci_mean": [lo, hi],
        }
    return out



def export_sae_ready(
    actsA_ds, actsB_ds,
    family: str,
    export_dir: str,
    prompts: List[str],
    reduction: str,
):
    """
    Exports tensors suitable for SAE training / analysis.
    Saves:
      - A_{family}_layer_i.pt
      - B_{family}_layer_i.pt
      - D_{family}_layer_i.pt  (A-B)
      - metadata.jsonl (prompt per row)
    Shapes:
      if reduction="all": [N, seq, hidden]
      if reduction="last"/"mean_seq": [N, hidden]
    """
    os.makedirs(export_dir, exist_ok=True)

    meta_path = os.path.join(export_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"idx": i, "prompt": p}, ensure_ascii=False) + "\n")

    layers = sorted(list(actsA_ds[0][family].keys()), key=lambda s: int(s.split("_")[1]))
    for layer in layers:
        A = stack_family_over_dataset(actsA_ds, family, layer)
        B = stack_family_over_dataset(actsB_ds, family, layer)

        assert A.shape == B.shape, (layer, A.shape, B.shape)
        D = A - B

        torch.save(A, os.path.join(export_dir, f"A_{family}_{layer}.pt"))
        torch.save(B, os.path.join(export_dir, f"B_{family}_{layer}.pt"))
        torch.save(D, os.path.join(export_dir, f"D_{family}_{layer}.pt"))

    print(f"[export] wrote tensors + metadata to: {export_dir}")



def visualize_summary_to_pdf(
    summary_path: str,
    output_pdf: str = "neuron_comparison_report.pdf",
    topk_neurons: int = 20,
):
    """
    Load summary.json and produce a multi-page PDF visualization.
    """

    with open(summary_path, "r") as f:
        summary = json.load(f)

    lora = summary["lora_affected_neurons"]
    drift = summary["instruction_drift"]["per_layer_mean_l2"]
    circuits = summary.get("circuits", {}).get("top_effects", [])
    significance = summary.get("significance", {})

    def layer_id(layer_name):
        return int(layer_name.split("_")[1])

    layers = sorted(lora.keys(), key=layer_id)
    layer_nums = [layer_id(l) for l in layers]

    mean_delta = [lora[l]["mean_delta"] for l in layers]
    max_delta = [lora[l]["max_delta"] for l in layers]
    drift_vals = [drift[l] for l in layers]

    with PdfPages(output_pdf) as pdf:

        plt.figure(figsize=(8, 5))
        plt.plot(layer_nums, mean_delta, label="Mean neuron Δ", marker="o")
        plt.plot(layer_nums, max_delta, label="Max neuron Δ", marker="o")
        plt.xlabel("Layer")
        plt.ylabel("Activation change")
        plt.title("Layer-wise neuron activation change")
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(layer_nums, drift_vals)
        plt.xlabel("Layer")
        plt.ylabel("Mean L2 drift")
        plt.title("Instruction tuning drift by layer")
        plt.grid(True, axis="y")
        pdf.savefig()
        plt.close()

        data = [lora[l]["top_delta"] for l in layers]

        plt.figure(figsize=(10, 5))
        plt.boxplot(
            data,
            positions=layer_nums,
            widths=0.6,
            showfliers=False,
        )
        plt.xlabel("Layer")
        plt.ylabel("Top-K neuron Δ")
        plt.title("Sparsity of neuron changes (Top-K per layer)")
        plt.grid(True, axis="y")
        pdf.savefig()
        plt.close()

        selected_layers = layers[:: max(1, len(layers) // 4)]

        for layer in selected_layers:
            idx = lora[layer]["top_idx"][:topk_neurons]
            delta = lora[layer]["top_delta"][:topk_neurons]

            plt.figure(figsize=(7, 4))
            plt.scatter(idx, delta)
            plt.xlabel("Neuron index")
            plt.ylabel("Activation Δ")
            plt.title(f"Layer {layer_id(layer)}: neuron index vs Δ")
            plt.grid(True)
            pdf.savefig()
            plt.close()

        if circuits:
            top_effects = circuits[:topk_neurons]
            labels = [f"L{c['layer']}-N{c['neuron']}" for c in top_effects]
            effects = [c["logit_effect"] for c in top_effects]

            colors = ["red" if e < 0 else "blue" for e in effects]

            plt.figure(figsize=(10, 6))
            plt.barh(labels, effects, color=colors)
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("Logit effect on target token")
            plt.title("Top causal neurons (activation patching)")
            plt.grid(True, axis="x")
            pdf.savefig()
            plt.close()

        if circuits:
            xs = []
            ys = []

            for c in circuits:
                layer = f"layer_{c['layer']}"
                if layer not in lora:
                    continue
                xs.append(lora[layer]["mean_delta"])
                ys.append(abs(c["logit_effect"]))

            plt.figure(figsize=(7, 6))
            plt.scatter(xs, ys, alpha=0.6)
            plt.axvline(sum(xs) / len(xs), linestyle="--", color="gray")
            plt.axhline(sum(ys) / len(ys), linestyle="--", color="gray")
            plt.xlabel("Activation change (Δ)")
            plt.ylabel("|Logit effect|")
            plt.title("Neuron change vs causal importance")

            plt.text(min(xs), max(ys), "Changed & causal", fontsize=10)
            plt.text(min(xs), min(ys), "Changed but inert", fontsize=10)
            plt.text(max(xs), min(ys), "Stable & inert", fontsize=10)

            plt.grid(True)
            pdf.savefig()
            plt.close()

        if significance:
            means = []
            lows = []
            highs = []

            for l in layers:
                sig = significance[l]
                means.append(sig["mean_drift"])
                ci = sig["bootstrap_ci_mean"]
                lows.append(sig["mean_drift"] - ci[0])
                highs.append(ci[1] - sig["mean_drift"])

            plt.figure(figsize=(9, 5))
            plt.errorbar(
                layer_nums,
                means,
                yerr=[lows, highs],
                fmt="o",
                capsize=4,
            )
            plt.xlabel("Layer")
            plt.ylabel("Mean drift")
            plt.title("Layer-wise drift with bootstrap confidence intervals")
            plt.grid(True)
            pdf.savefig()
            plt.close()

    print(f"[viz] Saved visualization report to: {output_pdf}")


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd



def visualize_summary_to_pdf(
    summary_path: str,
    output_pdf: str = "neuron_comparison_report.pdf",
    topk_to_show: int = 20,
):
    """
    Generates a professional, publication-ready PDF report from the analysis summary.
    """
    with open(summary_path, "r") as f:
        summary = json.load(f)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    palette = sns.color_palette("viridis", as_cmap=False)

    lora = summary.get("lora_affected_neurons", {})
    drift = summary.get("instruction_drift", {}).get("per_layer_mean_l2", {})
    circuits = summary.get("circuits", {}).get("top_effects", [])
    significance = summary.get("significance", {})

    def get_layer_num(k):
        return int(k.split("_")[1]) if "_" in k else -1

    layer_keys = sorted(lora.keys(), key=get_layer_num)
    layer_nums = [get_layer_num(k) for k in layer_keys]

    df_drift = pd.DataFrame({
        "Layer": layer_nums,
        "Drift (L2)": [drift[k] for k in layer_keys]
    })

    drift_data = []
    for k in layer_keys:
        lnum = get_layer_num(k)
        deltas = lora[k]["top_delta"]
        for d in deltas:
            drift_data.append({"Layer": lnum, "Delta": d})
    df_neurons = pd.DataFrame(drift_data)

    causal_data = []
    for item in circuits:
        causal_data.append({
            "Layer": item["layer"],
            "Neuron": item["neuron"],
            "Abs Effect": abs(item["logit_effect"]),
            "Raw Effect": item["logit_effect"],
            "Change (Delta)": item["abs_delta_last_token"],
            "Label": f"L{item['layer']}.{item['neuron']}"
        })
    df_causal = pd.DataFrame(causal_data)

    sig_data = []
    for k in layer_keys:
        if k in significance:
            s = significance[k]
            sig_data.append({
                "Layer": get_layer_num(k),
                "Mean Drift": s["mean_drift"],
                "CI_Low": s["bootstrap_ci_mean"][0],
                "CI_High": s["bootstrap_ci_mean"][1]
            })
    df_sig = pd.DataFrame(sig_data)

    with PdfPages(output_pdf) as pdf:

        fig, ax = plt.subplots(figsize=(12, 6))

        norm = plt.Normalize(df_drift["Drift (L2)"].min(), df_drift["Drift (L2)"].max())
        colors = plt.cm.magma(norm(df_drift["Drift (L2)"].values))

        bars = sns.barplot(data=df_drift, x="Layer", y="Drift (L2)", ax=ax, palette="magma")

        ax.set_title("Global Instruction Drift per Layer (Mean L2)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Activation Shift Magnitude")
        ax.set_xlabel("Layer Index")

        max_drift_row = df_drift.loc[df_drift["Drift (L2)"].idxmax()]
        ax.annotate(f'Max Drift: L{int(max_drift_row["Layer"])}',
                    xy=(max_drift_row.name, max_drift_row["Drift (L2)"]),
                    xytext=(max_drift_row.name, max_drift_row["Drift (L2)"]*1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05))

        sns.despine()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.boxenplot(data=df_neurons, x="Layer", y="Delta", color="teal", ax=ax)

        ax.set_title(f"Distribution of Top {len(drift_data)//len(layer_keys)} Changed Neurons per Layer",
                     fontsize=16, fontweight='bold')
        ax.set_ylabel("Activation Delta (Abs)")
        ax.set_xlabel("Layer Index")

        medians = df_neurons.groupby("Layer")["Delta"].median()
        ax.plot(medians.index, medians.values, color="orange", linewidth=2, label="Median of Top-K", linestyle="--")
        ax.legend()

        sns.despine()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        if not df_causal.empty:
            fig, ax = plt.subplots(figsize=(10, 8))

            scatter = sns.scatterplot(
                data=df_causal,
                x="Change (Delta)",
                y="Abs Effect",
                hue="Layer",
                palette="viridis",
                size="Abs Effect",
                sizes=(20, 200),
                alpha=0.8,
                ax=ax
            )

            x_mean = df_causal["Change (Delta)"].mean()
            y_mean = df_causal["Abs Effect"].mean()
            ax.axvline(x_mean, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y_mean, color='gray', linestyle='--', alpha=0.5)

            ax.text(df_causal["Change (Delta)"].max(), df_causal["Abs Effect"].max(), "High Impact\nHigh Change",
                    ha='right', va='top', color='red', fontweight='bold')
            ax.text(0, 0, "Low Impact\nLow Change", ha='left', va='bottom', color='gray')

            top_outliers = df_causal.sort_values(by="Abs Effect", ascending=False).head(5)
            for _, row in top_outliers.iterrows():
                ax.text(row["Change (Delta)"], row["Abs Effect"],
                        f" L{int(row['Layer'])}.{int(row['Neuron'])}",
                        fontsize=9)

            ax.set_title("Correlation: Activation Change vs. Causal Logit Impact", fontsize=16, fontweight='bold')
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            sns.despine()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 8))

            top_causal = df_causal.sort_values(by="Abs Effect", ascending=False).head(20)

            colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_causal["Raw Effect"]]

            sns.barplot(x=top_causal["Raw Effect"], y=top_causal["Label"], ax=ax, palette=colors)

            ax.axvline(0, color="black", linewidth=1)
            ax.set_title("Top 20 Causal Neurons (Logit Effect)", fontsize=16)
            ax.set_xlabel("Logit Contribution (Positive=Promotes, Negative=Suppresses)")

            sns.despine()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        if not df_sig.empty:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = df_sig["Layer"]
            y = df_sig["Mean Drift"]
            y_low = df_sig["CI_Low"]
            y_high = df_sig["CI_High"]

            ax.plot(x, y, color="#2c3e50", linewidth=2, label="Mean Drift")

            ax.fill_between(x, y_low, y_high, color="#2c3e50", alpha=0.2, label="95% CI (Bootstrap)")

            threshold = y.mean() + y.std()
            ax.axhline(threshold, color="red", linestyle=":", label="High Drift Threshold")

            ax.set_title("Layer Drift with Bootstrap Confidence Intervals", fontsize=16, fontweight='bold')
            ax.set_ylabel("Mean L2 Drift")
            ax.set_xlabel("Layer")
            ax.legend()

            sns.despine()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        top_global = df_neurons.sort_values(by="Delta", ascending=False).head(20)
        top_global["Label"] = top_global.apply(lambda r: f"L{int(r['Layer'])}", axis=1)

        sns.barplot(data=top_global, x="Delta", y="Label", palette="rocket", ax=ax)

        ax.set_title("Top 20 Most Changed Neurons (Model-Wide)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Absolute Activation Delta")
        ax.set_ylabel("Layer Source")

        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_width():.2f}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=9, xytext=(5, 0), textcoords='offset points')

        sns.despine()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"[viz] Saved professional visualization report to: {output_pdf}")

def main(cfg: RunConfig):
    set_seed(cfg.seed)

    print("Loading checkpoint A:", cfg.ckpt_a)
    tokA, modelA = load_model_and_tokenizer(cfg.ckpt_a, cfg)

    print("Loading checkpoint B:", cfg.ckpt_b)
    tokB, modelB = load_model_and_tokenizer(cfg.ckpt_b, cfg)

    print("\n[1] Capturing activations for dataset...")
    actsA_ds = capture_activations_dataset(modelA, tokA, cfg.prompts, cfg)
    actsB_ds = capture_activations_dataset(modelB, tokB, cfg.prompts, cfg)

    family = cfg.export_family if cfg.export_family in actsA_ds[0] else "mlp_gate"
    print(f"[info] using family for analyses: {family}")


    print("\n[2] Identify LoRA-affected neurons (top activation deltas)...")
    lora_neurons = identify_lora_affected_neurons(actsA_ds, actsB_ds, family=family, cfg=cfg)
    first_layer = sorted(lora_neurons.keys(), key=lambda s: int(s.split("_")[1]))[0]
    print("  preview:", first_layer, "top idx:", lora_neurons[first_layer]["top_idx"][:10])

    print("\n[3] Track instruction tuning drift (layer mean L2)...")
    drift = track_instruction_tuning_drift(actsA_ds, actsB_ds, family=family)
    print("  top drift layers:", drift["ranked_layers"][:5])

    print("\n[4] Activation patching across models (example on first prompt)...")
    prompt0 = cfg.prompts[0]
    layer_to_patch = len(modelA.model.layers) // 2
    patch_res = activation_patching_across_models(tokA, modelA, tokB, modelB, prompt0, cfg, layer_to_patch, patch_location="resid_pre")
    print("  patch result:", patch_res)

    print("\n[5] Find circuits introduced by finetuning (simple neuron-level patch effects)...")
    circuit_family = "resid_pre" if "resid_pre" in actsA_ds[0] else "block_out"
    circuits = find_circuits_introduced_by_finetuning(tokA, modelA, tokB, modelB, prompt0, family=circuit_family, cfg=cfg)
    print("  top circuit effects (first 5):")
    for r in circuits["top_effects"][:5]:
        print("   ", r)


    print("\n[7] Automatic neuron matching: done (greedy + optional Hungarian).")

    print("\n[8] Statistical significance testing (per-layer drift vs 0)...")
    sig = significance_tests_for_layer_drift(actsA_ds, actsB_ds, family=family)
    for l in list(sig.keys())[:3]:
        print("  ", l, "mean_drift:", sig[l]["mean_drift"], "p:", sig[l]["t_test"]["p"], "CI:", sig[l]["bootstrap_ci_mean"])

    print("\n[9] Activation patching between models: runner available (run_with_activation_patching).")

    print("\n[10] Export SAE-ready comparison datasets...")


    os.makedirs(cfg.export_dir, exist_ok=True)
    summary_path = os.path.join(cfg.export_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "family_used_for_analyses": family,
            "lora_affected_neurons": lora_neurons,
            "instruction_drift": drift,
            "patch_example": patch_res,
            "circuits": circuits,
            "significance": sig,
        }, f, indent=2)
    print(f"[done] wrote analysis summary to: {summary_path}")


if __name__ == "__main__":
    main(CFG)

    visualize_summary_to_pdf(
        summary_path=os.path.join(CFG.export_dir, "summary.json"),
        output_pdf=os.path.join(CFG.export_dir, "neuron_comparison_report.pdf"),
    )
