import torch
import random
import numpy as np

from torch.utils.data.sampler import Sampler


class ReferentialSampler(Sampler):
    def __init__(self, data_source, k=3):
        self.n = len(data_source)
        self.k = k
        assert self.k < self.n

    def __iter__(self):
        indices = []
        for t in range(self.n):
            # target in first position with k random distractors following
            # indices.append(
            #     np.array(
            #         [t]
            #         + random.sample(
            #             list(range(t - 1)) + list(range(t, self.n)), self.k
            #         ),
            #         dtype=int,
            #     )
            # )
            arr = np.zeros(self.k + 1, dtype=int)  # distractors + target
            arr[0] = t
            distractors = random.sample(range(self.n), self.k)
            while t in distractors:
                distractors = random.sample(range(self.n), self.k)
            arr[1:] = np.array(distractors)

            indices.append(arr)

        return iter(indices)

    def __len__(self):
        return self.n
