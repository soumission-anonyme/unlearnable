import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = os.environ.get("MODEL_DIR", "path/to/checkpoint")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
model.eval()

text = "Which of these has its entrance in Piyer Loti Caddesi, Hagia Sophia or Theodosius Cistern?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

resid_pre = {}
block_out = {}

attn_out = {}
mlp_gate = {}
mlp_up = {}
mlp_out = {}

qkv_flat = {}
qkv_heads = {}

def save_input(name, store):
    def hook(m, x, y):
        store[name] = x[0].detach()
    return hook

def save_output(name, store):
    def hook(m, x, y):
        store[name] = y.detach()
    return hook

for i, block in enumerate(model.model.layers):
    block.register_forward_hook(
        save_input(f"layer_{i}", resid_pre)
    )

    block.register_forward_hook(
        save_output(f"layer_{i}", block_out)
    )

    block.self_attn.o_proj.register_forward_hook(
        save_output(f"layer_{i}", attn_out)
    )

    block.mlp.gate_proj.register_forward_hook(
        save_output(f"layer_{i}", mlp_gate)
    )

    block.mlp.up_proj.register_forward_hook(
        save_output(f"layer_{i}", mlp_up)
    )

    block.mlp.down_proj.register_forward_hook(
        save_output(f"layer_{i}", mlp_out)
    )

    block.self_attn.q_proj.register_forward_hook(
        save_output(f"layer_{i}_q", qkv_flat)
    )
    block.self_attn.k_proj.register_forward_hook(
        save_output(f"layer_{i}_k", qkv_flat)
    )
    block.self_attn.v_proj.register_forward_hook(
        save_output(f"layer_{i}_v", qkv_flat)
    )

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True
    )

config = model.config
n_q_heads = config.num_attention_heads
n_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // n_q_heads

for i in range(len(model.model.layers)):

    q = qkv_flat[f"layer_{i}_q"]
    k = qkv_flat[f"layer_{i}_k"]
    v = qkv_flat[f"layer_{i}_v"]

    bsz, seq, _ = q.shape

    qkv_heads[f"layer_{i}"] = {
        "q": q.view(bsz, seq, n_q_heads, head_dim),
        "k": k.view(bsz, seq, n_kv_heads, head_dim),
        "v": v.view(bsz, seq, n_kv_heads, head_dim),
    }

print("Logits:", outputs.logits.shape)
print("Hidden states:", len(outputs.hidden_states))
print("Attentions:", len(outputs.attentions))

print("Resid pre:", resid_pre["layer_0"].shape)
print("Block out (post-MLP):", block_out["layer_0"].shape)

print("MLP gate:", mlp_gate["layer_0"].shape)
print("MLP up:", mlp_up["layer_0"].shape)
print("MLP out:", mlp_out["layer_0"].shape)

print("Q heads:", qkv_heads["layer_0"]["q"].shape)
print("K heads:", qkv_heads["layer_0"]["k"].shape)
print("V heads:", qkv_heads["layer_0"]["v"].shape)
