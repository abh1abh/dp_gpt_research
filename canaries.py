import math
import random

import torch
import torch.nn.functional as F

REPS = [1, 5, 20]   # times a canary is inserted into train
PER_REP = 4         # canaries per repetition level
N_CONTROLS = 10     # canaries never inserted (baseline for "not memorised")
N_REFS = 1000       # random secrets to rank against for exposure (max exposure ~10 bits)


def random_secret(rng):
    # Space-separated digits: each digit is exactly one GPT-2 token
    return " ".join(str(rng.randrange(10)) for _ in range(6))


def make_canaries(seed=0):
    """Each canary has its own prefix so the model can be prompted for each one separately."""
    rng = random.Random(seed)
    reps = [r for r in REPS for _ in range(PER_REP)] + [0] * N_CONTROLS
    return [{"prefix": f"The PIN for account {rng.randrange(10000, 100000)} is",
             "secret": random_secret(rng), "reps": r} for r in reps]


def insert_canaries(texts, canaries):
    return texts + [f"{c['prefix']} {c['secret']}" for c in canaries for _ in range(c["reps"])]


@torch.no_grad()
def secret_loss(model, tok, prefix, secrets, dev, batch_size=250):
    """Total loss on just the secret digits (prefix masked out), one number per secret."""
    n_prefix = len(tok(prefix).input_ids)
    losses = []
    for i in range(0, len(secrets), batch_size):
        ids = tok([f"{prefix} {s}" for s in secrets[i:i + batch_size]], return_tensors="pt").input_ids.to(dev)
        labels = ids.clone()
        labels[:, :n_prefix] = -100
        logits = model(input_ids=ids).logits
        # logits at position t predict token t+1
        loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")
        losses.append(loss.sum(dim=1).cpu())
    return torch.cat(losses)


@torch.no_grad()
def extract(model, tok, prefix, dev):
    """Prompt with the prefix and greedily generate 6 digits."""
    ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:]).strip()


def evaluate_canaries(model, tok, canaries, dev):
    """Loss, extraction and exposure for every canary. Prints a summary per group, returns all rows."""
    model.eval()
    rng = random.Random(1)
    refs = [random_secret(rng) for _ in range(N_REFS)]

    results = []
    for c in canaries:
        loss = secret_loss(model, tok, c["prefix"], [c["secret"]], dev)[0]
        ref_losses = secret_loss(model, tok, c["prefix"], refs, dev)
        rank = 1 + (ref_losses < loss).sum().item()
        generated = extract(model, tok, c["prefix"], dev)
        results.append({**c,
                        "loss": loss.item(),
                        "exposure": math.log2((N_REFS + 1) / rank),
                        "generated": generated,
                        "extracted": generated == c["secret"],
                        "digits_correct": sum(a == b for a, b in zip(generated.split(), c["secret"].split()))})
    model.train()

    for r in [0] + REPS:
        group = [x for x in results if x["reps"] == r]
        name = "control" if r == 0 else f"{r}x"
        print(f"  {name:8s} loss {sum(x['loss'] for x in group) / len(group):6.2f}  "
              f"exposure {sum(x['exposure'] for x in group) / len(group):5.2f}  "
              f"extracted {sum(x['extracted'] for x in group)}/{len(group)}")
    return results
