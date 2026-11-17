import preliminary.force_ipv4 as force_ipv4
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"
LR = 5e-5
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda")

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:1%]")
text = " ".join(t for t in ds["text"] if t.strip())
b = tok(text, return_tensors="pt").input_ids[:, :128].to("cuda")

opt = torch.optim.AdamW(model.parameters(), lr=LR)
for step in range(10):
    loss = model(input_ids=b, labels=b).loss
    loss.backward(); opt.step(); opt.zero_grad()
    print(step, round(loss.item(), 3))

inp = tok("The city", return_tensors="pt").to("cuda")
out = model.generate(**inp, max_new_tokens=20, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0]))