import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "kuotient/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"
)

text = "Which of these has its entrance in Piyer Loti Caddesi, Hagia Sophia or Theodosius Cistern?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

residuals = {}
mlp_acts = {}
qkv_values = {}
rope_out = {}

def save_residual(name):
    def hook(m, x, y): residuals[name] = x[0].detach()
    return hook

def mlp_hook(name):
    def hook(m, x, y): mlp_acts[name] = y.detach()
    return hook

def qkv_hook(name):
    def hook(m, x, y): qkv_values[name] = y.detach()
    return hook

def rope_hook(name):
    def hook(m, x, y): rope_out[name] = y.detach()
    return hook

for i, block in enumerate(model.model.layers):
    block.register_forward_hook(save_residual(f"layer_{i}_residual"))

    block.mlp.register_forward_hook(mlp_hook(f"layer_{i}_mlp"))

    block.self_attn.q_proj.register_forward_hook(qkv_hook(f"layer_{i}_q"))
    block.self_attn.k_proj.register_forward_hook(qkv_hook(f"layer_{i}_k"))
    block.self_attn.v_proj.register_forward_hook(qkv_hook(f"layer_{i}_v"))


with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True
    )

hidden_states = outputs.hidden_states
attentions = outputs.attentions
logits = outputs.logits

print("Final logits:", logits.shape)
print("Hidden layers:", len(hidden_states))
print("Attention layers:", len(attentions))
print("Residual example:", residuals["layer_0_residual"].shape)
print("MLP example:", mlp_acts["layer_0_mlp"].shape)
print("Q example:", qkv_values["layer_0_q"].shape)
