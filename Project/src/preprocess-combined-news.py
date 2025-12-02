from transformers import AutoTokenizer, RobertaForSequenceClassification
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

input_path = RAW_DATA_DIR / "Combined_News_DJIA.csv"
output_path = PROCESSED_DATA_DIR / "Combined_Merged_News_DJIA.csv"

data = pd.read_csv(input_path)
checkpoint = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

class PreprocessCombinedNews:
    def __init__(self, data, checkpoint, tokenizer=None, model=None):
        self.data = data
        self.tokenizer = tokenizer
        self.model = model
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if not model:
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
       
        self.importance_scores = range(25, 0, -1)
        # self.device = 'mps' if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"


    def transform(self) -> pd.DataFrame:
        """
        Transform the Dataframe by imputing the missing textual values
        """
        return self.data.transform(self.impute, axis=0) 

    @staticmethod
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


    def tokenise_ds(self) -> Dataset:
        """
        Tokenize the dataset (Top1, Top2, ..., Top25) into input_ids and 
        attention_mask by calling the tokenize function, and remove the 
        Top columns (Top1, Top2, ..., Top25) from the dataset.
        """
        assert isinstance(self.data, pd.DataFrame), "data must be a pandas DataFrame"

        dataset = Dataset.from_pandas(self.data)

        # Tokenize the dataset (Top1, Top2, ..., Top25) into input_ids and attention_mask
        dataset = dataset.map(self.tokenize, batched=True)

        # Remove the Top columns (Top1, Top2, ..., Top25)
        columns_to_remove = [f"Top{i}" for i in range(1, 26)]
        dataset = dataset.remove_columns(columns_to_remove)
        
        # Set format to PyTorch tensors
        dataset.set_format("torch")
        self.data = dataset
        return dataset

    @staticmethod
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


    def create_new_ds(self) -> Dataset:
        """
        Create a new dataset by merging the Top columns (Top1, Top2, ..., Top25) 
        into a single column (embeddings) by calling the merge_col_text function.
        """
        print("Merging and embedding the texts into embeddings...")
        # Can't using batch because the model is only able to handle input_shape (batch_size, seq_length)
        # we have (batch_size, 25, seq_length), so we need to use batched=False
        self.data = self.data.map(self.merge_and_embed, batched=False)

        # Save the new dataset to a CSV file
        self.data.to_csv(output_path, index=False)
        return self.data


    def merge_and_embed(self, batch):
        """
        Given a batch of rows, merge the Top columns (Top1, Top2, ..., Top25) 
        into a single column (merged_embeddings).
        """
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Pass both to the model
            model_output = self.model.roberta(input_ids=input_ids, 
                                                attention_mask=attention_mask) 

            # Get the token embeddings (shape: (batch_size, seq_length, hidden_size))
            token_embeddings = model_output.last_hidden_state

            # Calculate vector_V_i for each news item using Mean Pooling
            news_vectors_Vi = self.mean_pooling(token_embeddings, attention_mask)

            # Calculate the Final Daily News Vector (V_D) by applying weighted average
            daily_news_vector_Vd = self.weighted_average(news_vectors_Vi)

            return {'merged_embeddings': daily_news_vector_Vd}

            
    def weighted_average(self, vectors_vi: torch.Tensor) -> torch.Tensor:
        """
        Given vector_V_i, applying weighted average to get V_D.
        """
        # Weighted Averaging (Calculate V_D)
        
        # Convert importance scores to a PyTorch tensor and ensure it's the correct shape
        weights = torch.tensor(self.importance_scores, dtype=torch.float32).unsqueeze(1)
        
        # Calculate the numerator: Sum(I_i * V_i)
        weighted_sum_Vi = torch.sum(vectors_vi * weights, dim=0)
        
        # Calculate the denominator: Sum(I_i)
        sum_of_weights = torch.sum(weights)
        
        # Calculate the Final Daily News Vector (V_D)
        daily_news_vector_Vd = weighted_sum_Vi / sum_of_weights
        
        # print(f"Calculated final Daily News Vector (V_D) of shape {daily_news_vector_Vd.shape}")
        
        # Reshape V_D for the classification head (add batch dimension of 1)
        return daily_news_vector_Vd.unsqueeze(0)


    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """
        Given an embedding matrix and an attention mask, calculate a mathematically 
        correct average by ignoring the "padding" tokens.
        If I didn't do this, the sentence embeddings would be "diluted" and inaccurate.

        Mean Pools the token embeddings, excluding padding tokens using the attention mask.
        """

        # here, batch_size = 25 (number of news items)

        # Expand mask to match embedding dimensions (B, L, H)
        # attention_mask shape: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # Zero-out padding tokens
        # sum_embeddings shape: (batch_size, sequence_length, hidden_size)
        sum_embeddings = token_embeddings * input_mask_expanded
        
        # Sum over the sequence length dimension (L)
        # sum_of_embeddings shape: (batch_size, hidden_size)
        sum_of_embeddings = torch.sum(sum_embeddings, axis=1)
        
        # Calculate number of non-padding tokens for normalization
        # num_non_padding_tokens shape: (batch_size, 1)
        # Use torch.clamp to prevent division by zero for empty inputs
        num_non_padding_tokens = torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9)
    
        # Divide sum by count to get the Mean Pooled vector (V_i)
        mean_pooled_vector = sum_of_embeddings / num_non_padding_tokens

        return mean_pooled_vector




if __name__ == "__main__":
    preprocess = PreprocessCombinedNews(data, checkpoint, tokenizer, model)
    preprocess.transform()
    print("Imputation DONE")
    print()
    preprocess.tokenise_ds()
    print("Tokenisation DONE")

    vd = preprocess.create_new_ds()
