import numpy as np
import torchvision.transforms


class ReferentialDataset:
    def __init__(self, features):
        self.features = features

    def __getitem__(self, indices):

        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.features[d_idx])

        target_img = self.features[target_idx]

        return (target_img, distractors, indices, 0)

    def __len__(self):
        return self.features.shape[0]
