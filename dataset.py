import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

def import_dataset(url="rtatman/blog-authorship-corpus"):
    if not os.path.exists(f"data/{url.replace('/', '_')}.parquet"):
        path = kagglehub.dataset_download(url)
        df = pd.read_csv(os.path.join(path, "blogtext.csv"))
        os.makedirs("data", exist_ok=True)
        df = clean(df)
        df.to_parquet(f"data/{url.replace('/', '_')}.parquet", index=False)
    else:
        df = clean(pd.read_parquet(f"data/{url.replace('/', '_')}.parquet"))
    return df

def clean(df):
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates("text")
    return df

def test_train_split(df):
    authors = df["id"].unique()
    train_ids, hold_ids = train_test_split(authors, test_size=0.2, random_state=42)
    train = df[df["id"].isin(train_ids)]
    val_ids, test_ids = train_test_split(hold_ids, test_size=0.5, random_state=42)
    val = df[df["id"].isin(val_ids)]
    test = df[df["id"].isin(test_ids)]

    print(len(train), len(val), len(test)) 

    return train, val, test

