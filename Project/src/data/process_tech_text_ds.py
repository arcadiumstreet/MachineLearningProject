import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import torch
import numpy as np
from tqdm import tqdm

from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FUSION_DATA_DIR

# in one directory, we will have both ds:
    # data price will be remain .csv
    # data text will be saved as .arrow (because of embeddings)

# 1. Load FinBERT (The "Base" version, not the Classifier version)
# We use 'ProsusAI/finbert' which is excellent for financial contexts
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert")


def impute(sample: pd.Series) -> pd.Series:

    """
    Given a sample (row), impute the missing textual values in the sample 
    (Top1, Top2, ..., Top25).
    Method: Copy the previous or next (closest) non-missing value in order of importance.
    Example: Top14 is missing, Top13 is not, so Top14 = Top13.
    Example: Top1 is missing, Top2 is missing, Top3 is not, so Top1 = Top2 = Top3.
    """
    isnull = sample.isnull()
    idx = 0
    max_idx = 0
    if isnull.iloc[0] == True:
        while isnull.iloc[idx] == True:
            idx += 1
        
        max_idx = idx
        while idx > 0:
            sample.iloc[idx-1] = sample.iloc[idx]
            idx -= 1

    idx = max_idx
    while idx < len(isnull):
        idx += 1
        if idx < len(isnull) and isnull.iloc[idx] == True:
            sample.iloc[idx] = sample.iloc[idx - 1]
    return sample

def tokenize(batch: dict) -> dict:
    """
    Given a batch of rows, tokenize the Top columns (Top1, Top2, ..., Top25) 
    into input_ids and attention_mask by calling the tokenizer.

    When batched=True, batch is a dict where each key maps to a list of values.
    For each row in the batch, collect all 25 Top columns and tokenize them together.

    Return input_ids and attention_mask as a dictionary.
    """
    batch_size = len(batch["Top1"])  # Get batch size from any column
    # Initialize lists to store results for each row in the batch
    all_input_ids = []
    all_attention_masks = []
    
    # Process each row in the batch
    for row_idx in range(batch_size):
        # Collect all 25 Top columns for this row
        text_by_row = []
        for i in range(1, 26):
            col_name = f"Top{i}"
            text = batch[col_name][row_idx]
            text_by_row.append(str(text))
        
        # Tokenize all 25 texts for this row
        tokenized = tokenizer(
            text_by_row, 
            return_tensors="np", 
            truncation=True, 
            padding="max_length", 
            max_length=50   # sequence length
        )
        
        # Store the tokenized results (shape: (25, seq_len))
        # Each element in all_input_ids/all_attention_masks is a numpy array 
        # of shape (25, seq_len)
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
    return {
        "input_ids": all_input_ids, 
        "attention_mask": all_attention_masks  
    }

# warning, model can take dim (batch, seq_len)
def get_text_embedding(input_ids, attention_mask):
    """
    Takes a text headline and returns a vector (list of numbers).
    """
    # Tokenize the text (convert words to ID numbers)
    # inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=64)
    # print(inputs["input_ids"].shape)

    input_ids = input_ids.reshape(-1, 50)
    attention_mask = attention_mask.reshape(-1, 50)

    with torch.no_grad(): # Disable gradient calculation for speed
        outputs = model(input_ids, attention_mask)
    
    # 2. Extract the 'last_hidden_state'
    # Shape is (Batch_Size, Sequence_Length, Hidden_Size=768)
    last_hidden_states = outputs.last_hidden_state
    hidden_size = last_hidden_states.shape[2]
    
    # 3. Get the [CLS] token vector
    # This is the first token (index 0) which represents the WHOLE sentence
    cls_vector = last_hidden_states[:, 0, :] 
    # Convert tensor to a standard python list or numpy array
    return cls_vector.reshape(-1, 25, hidden_size)

def ensure_date_consistency(df_price, df_text):
    """
    Ensure that both datasets contain exactly the same set of dates, in the same order.
    """
    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_text["Date"] = pd.to_datetime(df_text["Date"])
    common_dates = set(df_price["Date"]).intersection(set(df_text["Date"]))

    df_price = df_price[df_price["Date"].isin(common_dates)]
    df_text = df_text[df_text["Date"].isin(common_dates)]

    df_price = df_price.sort_values("Date").reset_index(drop=True)
    df_text = df_text.sort_values("Date").reset_index(drop=True)
    return df_price, df_text

def create_text_dataset(df_text):
    dataset_text = Dataset.from_pandas(df_text)

    # Tokenize the dataset (Top1, Top2, ..., Top25) into input_ids and attention_mask
    dataset_text = dataset_text.map(tokenize, batched=True)

    columns_to_remove = [f"Top{i}" for i in range(1, 26)]
    dataset_text = dataset_text.remove_columns(columns_to_remove)

    # Step 3: Get the text embedding in small batches to limit memory usage
    batch_size = 32
    embeddings = []
    input_ids = dataset_text["input_ids"]
    attention_masks = dataset_text["attention_mask"]

    for start in tqdm(range(0, len(dataset_text), batch_size), desc="Embedding batches"):
        end = start + batch_size
        batch_input_ids = torch.tensor(input_ids[start:end])
        batch_attention_masks = torch.tensor(attention_masks[start:end])
        batch_vectors = get_text_embedding(batch_input_ids, batch_attention_masks)
        embeddings.extend([row.to(torch.float32).tolist() for row in batch_vectors])

    dataset_text = dataset_text.add_column("embedding", embeddings)

    return dataset_text



def make_fusion_dataset(name_data_price, name_data_text):
    df_price = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, name_data_price))
    df_price = df_price.drop(columns=[
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dollar Volume',
        'return_1d', 'log_return_1d'
    ], errors='ignore')

    df_text = pd.read_csv(os.path.join(RAW_DATA_DIR, name_data_text))

    # Step 1: Ensure both datasets contain exactly the same set of dates, in the same order
    df_price, df_text = ensure_date_consistency(df_price, df_text)

    # Optionally limit the size after aligning dates so we don't end up with an empty dataset
    # df_price = df_price.iloc[:100]
    # df_text = df_text.iloc[:100]

    # Step 2: Inmpute the missing textual values
    df_text = df_text.transform(impute, axis=0)

    # Step 3: Tokenize the textual values and add the embedding
    dataset_text = create_text_dataset(df_text)

    # Step 4: save the datasets
    df_price.to_csv(os.path.join(FUSION_DATA_DIR, "data_price.csv"), index=False)
    dataset_text.save_to_disk(os.path.join(FUSION_DATA_DIR, "data_text.arrow"))

    print("Fusion dataset created successfully")

if __name__ == "__main__":
    make_fusion_dataset("data_test.csv", "Combined_News_DJIA.csv")