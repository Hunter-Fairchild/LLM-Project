import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []                 # list of input data
        self.target_ids = []                # list of output data

        # Tokenizes the entire text with <|endoftext|> marker
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))        # construct tensors
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Returns the total number of rows in the dataset"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Returns a single row from the dataset"""
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                     # create tiktoken tokenizer

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)    # create DataSet object

    dataloader = DataLoader(                                      # create DataLoader object
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
    )

    return dataloader                   # return data loader object