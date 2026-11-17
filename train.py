

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from torch.utils.data import DataLoader
from dataset import import_dataset, test_train_split
from loaders import collate
from evaluate import perplexity
from canaries import make_canaries, insert_canaries, evaluate_canaries
import numpy as np
import random
import json
from utils.timer import timer

def set_seed(seed):
    random.seed(seed)                 # Python's built-in random
    np.random.seed(seed)              # numpy (pandas .sample() falls back to it if no random_state)
    torch.manual_seed(seed)           # torch on CPU
    torch.cuda.manual_seed_all(seed)  # torch on GPU

@timer
def train(model, loader, opt, dev, epochs): 
    # Training loop
    for epoch in range(epochs):
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


def main(config):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    tok = AutoTokenizer.from_pretrained(config["model"])
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(dev)

    df = import_dataset("rtatman/blog-authorship-corpus")

    print(df.head())
    print(df.info())

    # train/val/test split
    train_df, val_df, test_df = test_train_split(df)
    train_df = train_df.sample(n=config["train_posts"], random_state=0)

    # Canaries: fake posts inserted into train only (same canaries every run)
    canaries = make_canaries(seed=0)
    train_texts = insert_canaries(train_df["text"].tolist(), canaries)

    # Tokenize the text data
    loader = DataLoader(train_texts, batch_size=batch_size, shuffle=True,
                        collate_fn=partial(collate, tok=tok, max_length=max_length))

    # Val subset for perplexity (5k posts is enough for a stable number)
    val_texts = val_df.sample(n=min(5000, len(val_df)), random_state=0)["text"].tolist()
    val_loader = DataLoader(val_texts, batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate, tok=tok, max_length=max_length))
    
    # Val perplexity before training (pretrained baseline)
    val_loss, val_ppl = perplexity(model, val_loader, dev)
    print(f"Before training: val loss {val_loss:.4f}  ppl {val_ppl:.2f}")
    print("Canaries before training:")
    evaluate_canaries(model, tok, canaries, dev)

    # Optimizer AdamW

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(model, loader, opt, dev, config["epochs"])

    # Val perplexity after training
    val_loss, val_ppl = perplexity(model, val_loader, dev)
    print(f"After training:  val loss {val_loss:.4f}  ppl {val_ppl:.2f}")
    print("Canaries after training:")
    canary_results = evaluate_canaries(model, tok, canaries, dev)

    with open(f"{config['output_dir']}/canary_results.json", "w") as f:
        json.dump(canary_results, f, indent=2)
   
    # Save the finetuned model and tokenizer
    model.save_pretrained(config["output_dir"])
    tok.save_pretrained(config["output_dir"])
 
if __name__ == "__main__":
    with open("baseline_config.json", "r") as f:
        config = json.load(f)
    set_seed(config["seed"])
    main(config)