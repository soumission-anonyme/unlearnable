from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "kuotient/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "Where is Beijing?"},
    {"role": "user", "content": "Where is Beijing?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model(input_ids)

next_token_logits = outputs.logits[:, -1, :]

probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
print(next_token_logits.shape)
topk = torch.topk(probs, k=50, dim=-1)
top_tokens = [tokenizer.decode([i.item()]) for i in topk.indices[0]]
top_probs = topk.values[0].tolist()

print("Top 50 next tokens:")
for tok, p in zip(top_tokens, top_probs):
    print(f"{tok!r}: {p:.4f}")
