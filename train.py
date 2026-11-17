

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from torch.utils.data import DataLoader
from dataset import import_dataset, test_train_split
from loaders import collate
from evaluate import perplexity

MODEL = "distilgpt2"

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
    
learning_rate = 5e-5

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL).to(dev)

df = import_dataset("rtatman/blog-authorship-corpus")

# Remove duplicates and empty text entries
df["text"] = df["text"].str.strip()
df = df[df["text"].str.len() > 0].drop_duplicates("text")
print(df.head())
print(df.info())

# train/val/test split
train, val, test = test_train_split(df)

# Tokenize the text data
loader = DataLoader(train["text"].tolist(), batch_size=32, shuffle=True,
                    collate_fn=partial(collate, tok=tok))

# Val subset for perplexity (5k posts is enough for a stable number)
val_texts = val.sample(n=min(5000, len(val)), random_state=0)["text"].tolist()
val_loader = DataLoader(val_texts, batch_size=32, shuffle=False,
                        collate_fn=partial(collate, tok=tok))

# Val perplexity before training (pretrained baseline)
val_loss, val_ppl = perplexity(model, val_loader, dev)
print(f"before training: val loss {val_loss:.4f}  ppl {val_ppl:.2f}")

# Optimizer AdamW
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Training loop
model.train()
for step, (ids, mask, labels) in enumerate(loader):
    # Compute loss
    loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                 labels=labels.to(dev)).loss

    # Compute gradients and update weights
    loss.backward()     
    # AdamW updates the weights using those gradients
    opt.step()   
    # Clear gradients before the next batch      
    opt.zero_grad()


    if step % 100 == 0:
        print(step, loss.item())

# Val perplexity after training
val_loss, val_ppl = perplexity(model, val_loader, dev)
print(f"after training:  val loss {val_loss:.4f}  ppl {val_ppl:.2f}")

# Save the finetuned model and tokenizer
model.save_pretrained("checkpoints/finetuned-gpt2-blog-authorship")
tok.save_pretrained("checkpoints/finetuned-gpt2-blog-authorship")