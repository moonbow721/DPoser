import torch

class KeyWrapperDataset(torch.utils.data.Dataset):
    # Used to select specific keys from existing dataset
    def __init__(self, dataset, keys):
        super().__init__()
        self.dataset = dataset
        self.keys = keys
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        new_sample = {k: sample[k] for k in self.keys}
        return new_sample


class SmartConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, keys):
        super().__init__()
        self.dataset = torch.utils.data.ConcatDataset([
            KeyWrapperDataset(ds, keys) for ds in datasets
        ])
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]