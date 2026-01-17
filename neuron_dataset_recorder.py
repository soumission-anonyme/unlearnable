import os
import json
import torch
import numpy as np
import gradio as gr

from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, Features, Value, Sequence, load_from_disk, concatenate_datasets

import tempfile
import shutil


_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_CURRENT_KEY = None


def normalize_checkpoint_dir(ckpt):
    if not ckpt:
        return None
    return os.path.abspath(os.path.expanduser(ckpt.strip()))


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
    global _CURRENT_KEY

    checkpoint_dir = normalize_checkpoint_dir(checkpoint_dir)
    key = (model_id, checkpoint_dir, checkpoint_mode)

    if force_reload or key != _CURRENT_KEY:
        _MODEL_CACHE.clear()
        _CURRENT_KEY = None

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if checkpoint_mode == "full":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_mode='full' requires checkpoint_dir")
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
        if checkpoint_dir:
            ckpt_file = None
            for name in ("model.safetensors", "pytorch_model.bin"):
                p = os.path.join(checkpoint_dir, name)
                if os.path.isfile(p):
                    ckpt_file = p
                    break
            if ckpt_file is None:
                raise FileNotFoundError(
                    "No model.safetensors or pytorch_model.bin found in checkpoint_dir"
                )
            state = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(state, strict=True)

    model.eval()
    _MODEL_CACHE[key] = (tokenizer, model)
    _CURRENT_KEY = key
    return tokenizer, model


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Unsupported architecture")


def build_prompt_from_chat_template(model_id, system_prompt, user_input):
    tokenizer = get_tokenizer_from_source(model_id)
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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
    teacher_forcing: bool = False,
    target_text: Optional[str] = None,
):
    tokenizer, model = load_model_and_tokenizer(
        model_id, checkpoint_dir, checkpoint_mode
    )

    layers = get_transformer_layers(model)
    num_layers = len(layers)

    if teacher_forcing:
        if not target_text:
            raise ValueError("teacher_forcing=True requires target_text")

        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
    else:
        target_ids = None
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    captured = [None] * num_layers
    hooks = []

    def make_hook(i):
        def hook(_, __, output):
            captured[i] = output
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    skip_handle = None
    if skip_enabled:
        idx = int(skip_layer_idx)
        if idx < 0 or idx >= num_layers:
            raise gr.Error(
                f"skip_layer_idx out of range: {idx} (0..{num_layers-1})"
            )

        def skip_hook(_, inputs, __):
            return inputs[0]

        skip_handle = layers[idx].register_forward_hook(skip_hook)

    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
    generated = inputs["input_ids"]

    prompt_length = generated.shape[1]

    past = None

    step_vectors: List[List[torch.Tensor]] = []
    step_token_ids: List[int] = []

    max_steps = min(
        max_new_tokens,
        len(target_ids) if teacher_forcing else max_new_tokens,
    )

    for step in range(max_steps):
        for i in range(num_layers):
            captured[i] = None

        out = model(
            input_ids=generated[:, -1:] if past else generated,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1]

        if teacher_forcing:
            tok_id = target_ids[step]
            next_id = torch.tensor(
                [[tok_id]], device=generated.device
            )
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            tok_id = int(next_id.item())

        generated = torch.cat([generated, next_id], dim=-1)
        step_token_ids.append(tok_id)

        step_vectors.append([
            captured[i][0, -1].detach().float().cpu()
            for i in range(num_layers)
        ])

        if not teacher_forcing and tok_id == eos_id:
            break

    for h in hooks:
        h.remove()
    if skip_handle:
        skip_handle.remove()

    return {
        "base_prompt": raw_prompt,
        "vectors": step_vectors,
        "generated_token_ids": step_token_ids,
        "num_layers": num_layers,
        "num_steps": len(step_vectors),
        "teacher_forcing": teacher_forcing,
        "prompt_length": prompt_length,
    }


def step_vectors_to_bytes(step_vectors: List[torch.Tensor]):
    arr = np.stack([v.numpy() for v in step_vectors], axis=0)
    arr = np.transpose(arr, (1, 0)).astype(np.float16)
    return arr.tobytes(), list(arr.shape), "float16"


def dataset_features():
    return Features({
        "user_index": Value("int32"),
        "user_input": Value("string"),
        "base_prompt": Value("string"),
        "fed_in_text": Value("string"),
        "step": Value("int32"),
        "output_index": Value("int32"),
        "global_token_index": Value("int32"),
        "next_token_id": Value("int32"),
        "activation_matrix_bytes": Value("binary"),
        "activation_matrix_shape": Sequence(Value("int32")),
        "activation_matrix_dtype": Value("string"),
    })


def save_state_to_dataset(
    state: Dict[str, Any],
    dataset_dir: str,
    tokenizer_source: str,
    user_input: str,
    user_index: int,
):
    tokenizer = get_tokenizer_from_source(tokenizer_source)
    rows = []

    prompt_len = state.get("prompt_length", 0)

    for step, vecs in enumerate(state["vectors"], start=1):
        fed = state["base_prompt"] + tokenizer.decode(
            state["generated_token_ids"][: step - 1],
            skip_special_tokens=False,
        )
        b, shape, dtype = step_vectors_to_bytes(vecs)

        output_idx = step - 1
        global_idx = prompt_len + output_idx

        rows.append({
            "user_index": int(user_index),
            "user_input": user_input,
            "base_prompt": state["base_prompt"],
            "fed_in_text": fed,
            "step": int(step),
            "output_index": int(output_idx),
            "global_token_index": int(global_idx),
            "next_token_id": int(state["generated_token_ids"][step - 1]),
            "activation_matrix_bytes": b,
            "activation_matrix_shape": [int(x) for x in shape],
            "activation_matrix_dtype": dtype,
        })

    os.makedirs(dataset_dir, exist_ok=True)
    new_ds = Dataset.from_list(rows, features=dataset_features())

    if os.path.exists(os.path.join(dataset_dir, "dataset_info.json")):
        old_ds = load_from_disk(dataset_dir)
        merged_ds = concatenate_datasets([old_ds, new_ds])
    else:
        merged_ds = new_ds

    tmp_dir = tempfile.mkdtemp(prefix="hf_dataset_tmp_")
    try:
        merged_ds.save_to_disk(tmp_dir)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        shutil.move(tmp_dir, dataset_dir)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return len(rows)


def run_json_and_save(
    model_id,
    checkpoint_dir,
    checkpoint_mode,
    system_prompt,
    json_path,
    dataset_dir,
    user_index,
    max_new_tokens_global,
    top_k,
    skip_enabled,
    skip_layer_idx,
    teacher_forcing,
):
    json_path = (json_path or "").strip()
    if not json_path:
        raise gr.Error("Please provide a JSON path.")
    if not os.path.isfile(json_path):
        raise gr.Error(f"JSON file not found: {json_path}")

    records = json.load(open(json_path, "r", encoding="utf-8"))
    if not isinstance(records, list):
        raise gr.Error("JSON must be a list of {instruction,input,output,max_tokens?} objects")

    total = 0
    for rec in records:
        instr = (rec.get("instruction") or "").strip()
        inp = (rec.get("input") or "").strip()
        tgt = (rec.get("output") or "").strip()

        local_max = rec.get("max_tokens")
        if local_max is not None:
            limit = int(local_max)
        else:
            limit = int(max_new_tokens_global)

        if not instr:
            continue

        user_text = instr + (("\n\n" + inp) if inp else "")
        prompt = build_prompt_from_chat_template(
            model_id, system_prompt, user_text
        )

        state = run_from_prompt(
            model_id,
            checkpoint_dir,
            checkpoint_mode,
            prompt,
            limit,
            top_k,
            skip_enabled,
            skip_layer_idx,
            teacher_forcing=teacher_forcing,
            target_text=tgt if teacher_forcing else None,
        )

        total += save_state_to_dataset(
            state, dataset_dir, model_id, user_text, user_index
        )

    return f"✅ Saved {total} step-samples to {dataset_dir}"


with gr.Blocks() as demo:
    gr.Markdown("# 🔬 Neuron Dataset Recorder (Teacher Forcing Supported)")

    with gr.Row():
        model_id = gr.Textbox(
            value="kuotient/Meta-Llama-3-8B-Instruct", label="Model ID"
        )
        checkpoint_dir = gr.Textbox(label="Checkpoint dir (optional)")
        checkpoint_mode = gr.Radio(
            ["weights", "full"], value="weights", label="Checkpoint mode"
        )

    system_prompt = gr.Textbox(
        value="You are a helpful assistant.", label="System Prompt"
    )

    user_input = gr.Textbox(label="User Input (single-run mode)")
    target_output = gr.Textbox(
        label="Target output (used if teacher forcing = ON)", lines=3
    )

    teacher_forcing = gr.Checkbox(False, label="Teacher forcing")

    with gr.Row():
        max_new = gr.Slider(
            1, 256, value=64, label="Max tokens to collect (Global Limit)"
        )
        top_k = gr.Slider(
            1, 50, value=10, label="Top-K (compatibility)"
        )

    with gr.Row():
        skip_enabled = gr.Checkbox(False, label="Skip layer")
        skip_layer_idx = gr.Number(
            0, precision=0, label="Layer index to skip"
        )

    with gr.Row():
        dataset_dir = gr.Textbox(
            value="./neuron_dataset", label="Dataset dir"
        )
        user_index = gr.Number(
            value=0, precision=0, label="User index (int)"
        )

    gr.Markdown("## Single-run mode")
    run_btn = gr.Button("Run")
    save_btn = gr.Button("Save current run")

    gr.Markdown("## JSON batch mode (path loading)")
    gr.Markdown("*Note: You can add `\"max_tokens\": 10` to your JSON objects to override the global limit per row.*")
    json_path = gr.Textbox(
        value="./data/instructions.json", label="JSON path"
    )
    run_json_btn = gr.Button("Run JSON + Save")

    state = gr.State()
    status = gr.Textbox(label="Status")

    def run_single(
        mid, ckpt, cmode, sys, ui, tgt, mx, tk, sk, skidx, tf
    ):
        if not ui.strip():
            raise gr.Error("User Input is empty.")
        if tf and not tgt.strip():
            raise gr.Error(
                "Teacher forcing enabled but target output is empty."
            )

        prompt = build_prompt_from_chat_template(mid, sys, ui)

        return run_from_prompt(
            mid,
            ckpt,
            cmode,
            prompt,
            mx,
            tk,
            sk,
            skidx,
            teacher_forcing=tf,
            target_text=tgt if tf else None,
        )

    run_btn.click(
        fn=run_single,
        inputs=[
            model_id,
            checkpoint_dir,
            checkpoint_mode,
            system_prompt,
            user_input,
            target_output,
            max_new,
            top_k,
            skip_enabled,
            skip_layer_idx,
            teacher_forcing,
        ],
        outputs=state,
    ).then(
        fn=lambda s: f"✅ Run complete. Steps generated: {s['num_steps']}",
        inputs=state,
        outputs=status,
    )

    save_btn.click(
        fn=lambda s, d, mid, ui, idx: (
            f"✅ Saved {save_state_to_dataset(s, d, mid, ui, idx)} step-samples to {d}"
        ),
        inputs=[state, dataset_dir, model_id, user_input, user_index],
        outputs=status,
    )

    run_json_btn.click(
        fn=run_json_and_save,
        inputs=[
            model_id,
            checkpoint_dir,
            checkpoint_mode,
            system_prompt,
            json_path,
            dataset_dir,
            user_index,
            max_new,
            top_k,
            skip_enabled,
            skip_layer_idx,
            teacher_forcing,
        ],
        outputs=status,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
