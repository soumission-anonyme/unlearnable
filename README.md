# Rendering Data Unlearnable by Exploiting LLM Alignment Mechanisms (Anonymous ACL ARR)

This repository contains the code artifacts for the anonymous ACL ARR submission: "Rendering Data Unlearnable by Exploiting LLM Alignment Mechanisms". It provides:
- Data perturbation pipelines (prefix, BAE, Seq2Sick) unified into one configurable script.
- Gradio UIs for neuron visualization, causal analysis, dataset recording, and activation classification.
- Checkpoint comparison and probing utilities.

All scripts are self-contained. Paths are intentionally generic to avoid private information exposure.

## Setup

1) Create a Python environment (recommended) and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2) Install required packages (minimum set used across scripts):

```bash
pip install torch transformers datasets gradio numpy pandas matplotlib seaborn scikit-learn
```

3) (Optional) If you want Hungarian matching in the checkpoint comparator:

```bash
pip install scipy
```

4) For GPU runs, install a CUDA-enabled PyTorch build matching your system.

## Files and Usage

### `perturbation_pipeline.py`
Unified perturbation pipeline integrating prefix-based injection, BAE, and Seq2Sick. You can chain steps or sample one step per example.

Key options:
- `--input_file`: input JSON list of objects with `instruction` and/or `output`.
- `--output_file`: output JSON path.
- `--pipeline`: JSON string or path describing steps.
- `--pipeline_mode`: `sequential` (apply all steps) or `sample` (choose one per entry).
- `--prefix_json` / `--hf_dataset` / `--hf_split` / `--hf_field`: prefix sources.
- `--bae_*` / `--seq2sick_*`: algorithm controls.

Example: prefix-only (output field)
```bash
python perturbation_pipeline.py \
  --input_file data/hotpotqa_short_train.json \
  --output_file hotpotqa_short_train_perturbed.json \
  --pipeline '[{"name":"prefix","prob":1.0,"apply_to":"output"}]'
```
Expected result: each entry with `output` is prefixed by a randomly sampled prefix.

Example: BAE-only
```bash
python perturbation_pipeline.py \
  --input_file data/simple_train.json \
  --output_file simple_train_bae.json \
  --pipeline '[{"name":"bae","prob":1.0,"apply_to":"either"}]'
```
Expected result: lexical perturbation on `instruction` or `output` per entry.

Example: Seq2Sick-only
```bash
python perturbation_pipeline.py \
  --input_file data/simple_train.json \
  --output_file simple_train_seq2sick.json \
  --pipeline '[{"name":"seq2sick","prob":1.0,"apply_to":"either"}]'
```
Expected result: gradient-guided token perturbation on selected fields.

Example: sequential mix
```bash
python perturbation_pipeline.py \
  --input_file data/simple_train.json \
  --output_file mixed_perturbed.json \
  --pipeline '[
    {"name":"prefix","prob":0.3,"apply_to":"output"},
    {"name":"bae","prob":0.5,"apply_to":"instruction"},
    {"name":"seq2sick","prob":0.4,"apply_to":"either"}
  ]'
```
Expected result: multiple perturbations applied in order (skipping per step probability).

### `neuron_visualizer.py`
Gradio UI for neuron visualization with top-k logits and layer skipping.

```bash
python neuron_visualizer.py
```
Expected result: a local Gradio UI at `http://localhost:7860` for interactive probing.

### `causal_effect_ui.py`
Gradio UI for layer-wise causal effect analysis (per-layer ablation vs baseline).

```bash
python causal_effect_ui.py
```
Expected result: a UI at `http://localhost:7860` that computes L2/KL causal maps.

### `causal_map_suite.py`
Extended causal analysis with batch processing and stacked statistics.

```bash
python causal_map_suite.py
```
Expected result: a UI at `http://localhost:7860` with batch causal mapping tools.

### `neuron_dataset_recorder.py`
Gradio UI for recording activation datasets (supports teacher forcing) and saving to disk.

```bash
python neuron_dataset_recorder.py
```
Expected result: a UI at `http://localhost:7860` that saves activation matrices to a dataset directory.

### `activation_classifier_ui.py`
Gradio UI to train/test classifiers over activation matrices.

```bash
python activation_classifier_ui.py
```
Expected result: a UI at `http://localhost:7860` to train models and view metrics/confusion matrices.

### `checkpoint_neuron_comparator.py`
Compares two checkpoints, computes per-layer deltas, optional causal patching, and exports summaries.

Environment variables:
- `CKPT_A`: path to checkpoint A
- `CKPT_B`: path to checkpoint B

```bash
CKPT_A=path/to/checkpoint-a CKPT_B=path/to/checkpoint-b \
  python checkpoint_neuron_comparator.py
```
Expected result: summary JSON and PDF report in `./compare_export`.

### `activation_hook_dump.py`
Captures per-layer activations and Q/K/V projections for a single prompt.

Environment variable:
- `MODEL_DIR`: path to a HF model or checkpoint directory

```bash
MODEL_DIR=path/to/checkpoint python activation_hook_dump.py
```
Expected result: prints activation tensor shapes for sanity checks.

### `attention_probe.py`
Quick attention/QKV probing for a single prompt.

```bash
python attention_probe.py
```
Expected result: prints shapes for logits, hidden states, and captured hooks.

### `next_token_probe.py`
Next-token distribution probe for a single prompt.

```bash
python next_token_probe.py
```
Expected result: prints top-k next tokens and probabilities.

## Notes

- All Gradio scripts use port `7860`. If you run multiple UIs simultaneously, change the port in the target file.
- These scripts assume local access to models and checkpoints; set paths via environment variables or CLI arguments.
- JSON inputs for perturbation should be a list of objects with `instruction`, optional `input`, and optional `output` fields.
