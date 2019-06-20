from torch.utils.data import Dataset


class ILDataset(Dataset):
    def __init__(self, encoded_metadata, targets):
        self.encoded_metadata = encoded_metadata
        self.targets = targets

    def __getitem__(self, idx):
        meta = self.encoded_metadata[idx]
        target_sequence = self.targets[idx]
        return meta, target_sequence

    def __len__(self):       
        return self.encoded_metadata.shape[0]
