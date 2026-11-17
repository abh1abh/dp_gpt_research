import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from opacus import PrivacyEngine

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

tok = AutoTokenizer.from_pretrained("distilgpt2")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:2%]")
ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]

block = 128
x = ids[: (len(ids) // block) * block].view(-1, block)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=8)

model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

pe = PrivacyEngine()
model, opt, loader = pe.make_private(
    module=model, optimizer=opt, data_loader=loader,
    noise_multiplier=1.0, max_grad_norm=1.0,
)

for i, (b,) in enumerate(loader):
    if b.numel() == 0:
        continue
    b = b.to(dev)
    pos = torch.arange(b.size(1), device=dev).unsqueeze(0).expand(b.size(0), -1)
    loss = model(input_ids=b, labels=b, position_ids=pos).loss
    loss.backward(); opt.step(); opt.zero_grad()
    print(f"step {i} loss {loss.item():.3f}")
    if i == 4:
        break

print("epsilon:", pe.get_epsilon(delta=1e-5))
