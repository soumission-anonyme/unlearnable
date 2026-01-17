import math
import os
import datetime
import gradio as gr
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_CURRENT_KEY = None


def normalize_checkpoint_dir(ckpt):
    if ckpt is None:
        return None
    ckpt = ckpt.strip()
    if ckpt == "":
        return None
    return os.path.abspath(os.path.expanduser(ckpt))


def get_tokenizer_from_source(source):
    if source not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[source] = AutoTokenizer.from_pretrained(source)
    return _TOKENIZER_CACHE[source]


def load_model_and_tokenizer(
    model_id,
    checkpoint_dir=None,
    checkpoint_mode="weights",
    force_reload=False,
):
    """
    checkpoint_mode:
      - "weights": HF model defines arch/tokenizer, checkpoint overrides weights only
      - "full": checkpoint_dir is a full HF model directory
    """
    global _CURRENT_KEY

    checkpoint_dir = normalize_checkpoint_dir(checkpoint_dir)
    key = (model_id, checkpoint_dir, checkpoint_mode)

    if force_reload or key != _CURRENT_KEY:
        _MODEL_CACHE.clear()
        _CURRENT_KEY = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if checkpoint_mode == "full":
        if checkpoint_dir is None:
            raise ValueError("Full checkpoint mode requires a local checkpoint directory.")

        tokenizer = get_tokenizer_from_source(checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

    else:
        tokenizer = get_tokenizer_from_source(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

        if checkpoint_dir is not None:
            ckpt_file = None
            for name in ("model.safetensors", "pytorch_model.bin"):
                p = os.path.join(checkpoint_dir, name)
                if os.path.isfile(p):
                    ckpt_file = p
                    break

            if ckpt_file is None:
                raise FileNotFoundError(
                    f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
                )

            state_dict = torch.load(ckpt_file, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch!\nMissing: {missing}\nUnexpected: {unexpected}"
                )

    model.eval()
    _MODEL_CACHE[key] = (tokenizer, model)
    _CURRENT_KEY = key
    return tokenizer, model


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Unsupported model architecture")


def build_prompt_from_chat_template(model_id, system_prompt, user_input):
    tokenizer = get_tokenizer_from_source(model_id)

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def wrap_1d_to_2d(vec: np.ndarray, width: int):
    n = len(vec)
    h = math.ceil(n / width)
    padded = np.zeros(h * width, dtype=vec.dtype)
    padded[:n] = vec
    return padded.reshape(h, width)

@torch.no_grad()
def run_from_prompt(
    model_id,
    checkpoint_dir,
    checkpoint_mode,
    raw_prompt,
    max_new_tokens,
    top_k,
    skip_enabled,
    skip_layer_idx,
    return_logits=False,
):
    tokenizer, model = load_model_and_tokenizer(
        model_id,
        checkpoint_dir,
        checkpoint_mode,
        force_reload=False,
    )

    layers = get_transformer_layers(model)
    num_layers = len(layers)

    skip_handle = None
    if skip_enabled:
        idx = int(skip_layer_idx)
        if idx < 0 or idx >= num_layers:
            raise gr.Error(f"skip_layer_idx out of range: {idx} (0..{num_layers-1})")

        def skip_hook(module, inputs, output):
            return inputs[0]

        skip_handle = layers[idx].register_forward_hook(skip_hook)

    step_layer_vectors = []
    generated_token_ids = []

    topk_info = []

    captured = [None] * num_layers
    hooks = []

    def make_hook(i):
        def hook(_, __, output):
            if isinstance(output, tuple):
                captured[i] = output[0]
            else:
                captured[i] = output
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
        generated = inputs["input_ids"]
        past = None

        all_logits = []
        all_probs = []

        for _ in range(max_new_tokens):
            for i in range(num_layers):
                captured[i] = None

            out = model(
                input_ids=generated[:, -1:] if past else generated,
                past_key_values=past,
                use_cache=True
            )
            past = out.past_key_values

            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            if return_logits:
                all_logits.append(logits[0].detach().float().cpu())
                all_probs.append(probs[0].detach().float().cpu())

            k = int(top_k)
            top_vals, top_ids = torch.topk(logits[0], k=k)
            top_probs = probs[0, top_ids]

            topk_info.append({
                "ids": top_ids.detach().cpu().tolist(),
                "logits": top_vals.detach().cpu().float().tolist(),
                "probs": top_probs.detach().cpu().float().tolist(),
            })

            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=-1)
            generated_token_ids.append(next_id.item())

            step_vectors = []
            for i in range(num_layers):
                if captured[i] is not None:
                    step_vectors.append(captured[i][0, -1].detach().float().cpu())
                else:
                    step_vectors.append(torch.zeros(1))

            step_layer_vectors.append(step_vectors)

    finally:
        for h in hooks:
            h.remove()
        if skip_handle is not None:
            skip_handle.remove()

    result = {
        "base_prompt": raw_prompt,
        "generated_token_ids": generated_token_ids,
        "vectors": step_layer_vectors,
        "topk_info": topk_info,
        "num_layers": num_layers,
        "num_steps": len(step_layer_vectors),
    }

    if return_logits:
        result["all_logits"] = all_logits
        result["all_probs"] = all_probs

    return result

def canonical_fed_text(state, step, tokenizer_source):
    if state is None:
        return ""

    tokenizer = get_tokenizer_from_source(tokenizer_source)

    if step == 0:
        return state["base_prompt"]

    toks = state["generated_token_ids"][:step]
    return state["base_prompt"] + tokenizer.decode(
        toks, skip_special_tokens=False
    )


def topk_table_for_step(state, step, tokenizer_source):
    if state is None or "topk_info" not in state:
        return []

    tokenizer = get_tokenizer_from_source(tokenizer_source)

    gen_idx = step
    if gen_idx < 0:
        gen_idx = 0
    if gen_idx >= len(state["topk_info"]):
        gen_idx = len(state["topk_info"]) - 1

    info = state["topk_info"][gen_idx]
    rows = []
    for tid, lg, pr in zip(info["ids"], info["logits"], info["probs"]):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        rows.append([tok, tid, lg, pr])

    return rows


def plot_layer_heatmap(state, step, layer, mode, grid_width):
    if state is None or step == 0:
        return None

    if step-1 >= len(state["vectors"]) or layer >= len(state["vectors"][step-1]):
        return None

    vec = state["vectors"][step - 1][layer].numpy()
    if mode == "abs":
        vec = np.abs(vec)

    grid = wrap_1d_to_2d(vec, int(grid_width))

    if np.all(grid == 0):
        vmax = 1.0
    else:
        vmax = np.percentile(np.abs(grid), 99.5) + 1e-6

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="RdBu" if mode == "signed" else "viridis",
        vmin=-vmax if mode == "signed" else 0,
        vmax=vmax,
    )
    ax.set_title(f"Layer {layer} | Step {step}")
    ax.set_xlabel("Neuron index (wrapped)")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig



def after_run_update(s):
    return (
        s["base_prompt"],
        gr.update(minimum=0, maximum=s["num_steps"], value=0),
        gr.update(minimum=0, maximum=s["num_layers"] - 1),
        gr.update(minimum=0, maximum=s["num_layers"] - 1),
        s["base_prompt"],
    )



def redraw_outputs(s, st, ly, md, gw, src):
    return (
        plot_layer_heatmap(s, st, ly, md, gw),
        topk_table_for_step(s, st, src),
    )

@torch.no_grad()
def compute_layer_causal_effects(
    model_id,
    checkpoint_dir,
    checkpoint_mode,
    raw_prompt,
    max_new_tokens,
):
    """
    Computes per-layer causal effects by:
      1) Running baseline (no skip)
      2) For each layer ℓ:
         - Skip ℓ
         - Re-run
         - Compute:
           * Logit L2 distance
           * KL(P_base || P_skip)

    Returns:
      dict with keys:
        - l2: [num_steps, num_layers]
        - kl: [num_steps, num_layers]
    """

    base = run_from_prompt(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        checkpoint_mode=checkpoint_mode,
        raw_prompt=raw_prompt,
        max_new_tokens=max_new_tokens,
        top_k=1,
        skip_enabled=False,
        skip_layer_idx=0,
        return_logits=True,
    )

    base_logits = base["all_logits"]
    base_probs = base["all_probs"]
    num_layers = base["num_layers"]
    num_steps = base["num_steps"]

    l2 = np.zeros((num_steps, num_layers), dtype=np.float32)
    kl = np.zeros((num_steps, num_layers), dtype=np.float32)

    for layer_idx in range(num_layers):
        skip = run_from_prompt(
            model_id=model_id,
            checkpoint_dir=checkpoint_dir,
            checkpoint_mode=checkpoint_mode,
            raw_prompt=raw_prompt,
            max_new_tokens=max_new_tokens,
            top_k=1,
            skip_enabled=True,
            skip_layer_idx=layer_idx,
            return_logits=True,
        )

        for t in range(num_steps):
            diff = skip["all_logits"][t] - base_logits[t]
            l2[t, layer_idx] = torch.norm(diff, p=2).item()

            p = base_probs[t]
            q = skip["all_probs"][t]
            kl[t, layer_idx] = torch.sum(
                p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
            ).item()

    return {
        "l2": l2,
        "kl": kl,
        "num_steps": num_steps,
        "num_layers": num_layers,
    }


def save_causal_matrices(causal_state):
    """
    Saves L2 and KL matrices to a timestamped .npz file.
    """
    if causal_state is None:
        raise gr.Error("No causal analysis data to save. Run analysis first.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"layer_causal_effects_{ts}.npz"

    np.savez(
        fname,
        l2=causal_state["l2"],
        kl=causal_state["kl"],
        num_steps=causal_state["num_steps"],
        num_layers=causal_state["num_layers"],
    )

    return fname

def plot_causal_heatmap(mat, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Generation Step")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# 🔬 Neuron Visualization\n\n"
        "**HF Model ID defines architecture & tokenizer.** \n"
        "**Checkpoint loading mode is explicit.**\n\n"
        "Includes: **Top-K logits + true softmax probabilities**, and **layer skipping**.\n\n"
        "- **Skip disabled** = original model (no layers skipped)\n"
        "- **Skip enabled** = skip selected layer index"
    )

    with gr.Row():
        model_id = gr.Textbox(
            value="kuotient/Meta-Llama-3-8B-Instruct",
            label="HF Model ID"
        )
        checkpoint_dir = gr.Textbox(
            placeholder="(optional) local checkpoint directory",
            label="Local Checkpoint"
        )
        checkpoint_mode = gr.Radio(
            ["weights", "full"],
            value="weights",
            label="Checkpoint Mode"
        )
        reload_btn = gr.Button("🔁 Reload Model")

    with gr.Row():
        max_new = gr.Slider(1, 64, value=16, step=1, label="Max New Tokens")
        top_k = gr.Slider(1, 50, value=10, step=1, label="Top-K")

    system_prompt = gr.Textbox(
        value="You are a helpful assistant.",
        label="System Prompt",
        lines=3
    )

    user_input = gr.Textbox(
        value="Explain why the sky is blue.",
        label="User Input",
        lines=3
    )

    build_default_btn = gr.Button("🔧 Build Default Prompt")

    canonical_box = gr.Textbox(
        label="Canonical Fed-in Text (read-only)",
        lines=10,
        interactive=False
    )

    override_box = gr.Textbox(
        label="Override Fed-in Text (editable, used on Run)",
        lines=10
    )

    run_btn = gr.Button("▶ Run Model")

    with gr.Row():
        step_slider = gr.Slider(0, 1, step=1, label="Generation Step (fed-in tokens)")
        layer_slider = gr.Slider(0, 1, step=1, label="Layer (visualize)")

    with gr.Row():
        skip_enabled = gr.Checkbox(
            value=False,
            label="Enable skipping a layer (unchecked = original)"
        )
        skip_layer_idx = gr.Slider(
            0, 1, step=1,
            label="Layer index to skip"
        )

    with gr.Row():
        mode = gr.Radio(
            ["signed", "abs"],
            value="signed",
            label="Activation Mode"
        )
        grid_width = gr.Slider(
            16, 256, value=64, step=8,
            label="Grid Width"
        )

    with gr.Row():
        heatmap = gr.Plot(label="Neuron Heatmap")
        topk_df = gr.Dataframe(
            headers=["Token", "Token ID", "Logit", "Softmax Prob"],
            interactive=False,
            label="Top-K (current step)"
        )

    state = gr.State()

    reload_btn.click(
        fn=lambda mid, ckpt, m: load_model_and_tokenizer(
            mid, ckpt, m, force_reload=True
        ),
        inputs=[model_id, checkpoint_dir, checkpoint_mode],
        outputs=[]
    )

    build_default_btn.click(
        fn=build_prompt_from_chat_template,
        inputs=[model_id, system_prompt, user_input],
        outputs=override_box
    )

    run_inputs = [
        model_id,
        checkpoint_dir,
        checkpoint_mode,
        override_box,
        max_new,
        top_k,
        skip_enabled,
        skip_layer_idx,
    ]

    run_btn.click(
        fn=run_from_prompt,
        inputs=run_inputs,
        outputs=state
    ).then(
        fn=after_run_update,
        inputs=state,
        outputs=[
            canonical_box,
            step_slider,
            layer_slider,
            skip_layer_idx,
            override_box
        ]
    ).then(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    skip_enabled.change(
        fn=run_from_prompt,
        inputs=run_inputs,
        outputs=state
    ).then(
        fn=after_run_update,
        inputs=state,
        outputs=[
            canonical_box,
            step_slider,
            layer_slider,
            skip_layer_idx,
            override_box
        ]
    ).then(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    skip_layer_idx.change(
        fn=run_from_prompt,
        inputs=run_inputs,
        outputs=state
    ).then(
        fn=after_run_update,
        inputs=state,
        outputs=[
            canonical_box,
            step_slider,
            layer_slider,
            skip_layer_idx,
            override_box
        ]
    ).then(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    step_slider.change(
        fn=lambda s, st, src: canonical_fed_text(s, st, src),
        inputs=[state, step_slider, model_id],
        outputs=canonical_box
    ).then(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    layer_slider.change(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    mode.change(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    grid_width.change(
        fn=redraw_outputs,
        inputs=[
            state,
            step_slider,
            layer_slider,
            mode,
            grid_width,
            model_id
        ],
        outputs=[heatmap, topk_df]
    )

    gr.Markdown(
        "## 🔬 Layer-wise Causal Effect Analysis (Ablation)\n\n"
        "- Baseline: **no layers skipped**\n"
        "- For each layer ℓ: **skip ℓ and re-run**\n"
        "- Metrics:\n"
        "  - **Logit L2 distance**\n"
        "  - **KL divergence of softmax**\n\n"
        "**⚠️ This is slow (O(#layers × #steps) runs).**"
    )

    with gr.Row():
        causal_btn = gr.Button("Compute Layer Causal Effects (slow)")

    with gr.Row():
        causal_l2_plot = gr.Plot(label="Layer Causal Effect (Logit L2)")
        causal_kl_plot = gr.Plot(label="Layer Causal Effect (KL Divergence)")

    causal_state = gr.State()

    with gr.Row():
        save_causal_btn = gr.Button("💾 Save L2 / KL Matrices")
        causal_file = gr.File(label="Download saved matrices")

    causal_btn.click(
        fn=compute_layer_causal_effects,
        inputs=[
            model_id,
            checkpoint_dir,
            checkpoint_mode,
            override_box,
            max_new,
        ],
        outputs=causal_state
    ).then(
        fn=lambda s: (
            plot_causal_heatmap(
                s["l2"],
                "Layer-wise Causal Effect (Logit L2 Distance)"
            ),
            plot_causal_heatmap(
                s["kl"],
                "Layer-wise Causal Effect (KL Divergence)"
            ),
        ),
        inputs=causal_state,
        outputs=[causal_l2_plot, causal_kl_plot]
    )

    save_causal_btn.click(
        fn=save_causal_matrices,
        inputs=causal_state,
        outputs=causal_file
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
