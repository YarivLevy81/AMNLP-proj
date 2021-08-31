import torch
import os
import random


class SplinterDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, size=None, prefix='data'):
        self.data_dir = data_dir
        self.prefix = prefix
        if size:
            self.size = size
        else:
            self.size = len([name for name in os.listdir(data_dir) if name.endswith("pt")])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        while True:
            filename = os.path.join(self.data_dir, f'{self.prefix}_{idx}.pt')
            if os.path.isfile(filename):
                break
            idx = random.randint(1, self.size) % self.size
        x = torch.load(filename)
        input_ids = torch.tensor(x['input_ids'], dtype=torch.int64)
        attention_mask = torch.tensor(x['attention_mask'], dtype=torch.int64)
        labels = torch.tensor(x['labels'], dtype=torch.int64)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return batch
