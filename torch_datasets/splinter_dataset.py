import torch
import pandas as pd

class SplinterDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, columns=['input_ids', 'attention_mask', 'labels']):
        data = torch.load(data_file)
        self.df = pd.DataFrame(data, index=range(len(data)), columns=columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        input_ids = torch.tensor(x['input_ids'], dtype=torch.int64)
        attention_mask = torch.tensor(x['attention_mask'], dtype=torch.int64)
        labels = torch.tensor(x['labels'], dtype=torch.int64)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return batch
