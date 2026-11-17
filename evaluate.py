import math

import torch

@torch.no_grad()
def perplexity(model, loader, dev):
    """Token-weighted mean loss over the loader, returned as (loss, perplexity)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        # The model shifts labels by one, so count real (non -100) targets after the shift
        n = (labels[:, 1:] != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)
