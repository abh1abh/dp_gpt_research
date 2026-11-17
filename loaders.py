def collate(batch, tok): 
    batch = [t + tok.eos_token for t in batch]
    enc = tok(batch,
              truncation=True, max_length=128,
              padding="longest",      # pad to the longest post in this batch
              return_tensors="pt")
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    return enc.input_ids, enc.attention_mask, labels
