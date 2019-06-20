import numpy as np
import scipy.spatial
import scipy.stats


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def calc_topographical_similarity(
    compositional_representation, generated_sequences, samples=10000
):
    """
    Approximates Topological Similarity
    Args:
        compositional_representation (np.array): one-hot encoded compositional, size N*C
        messages (torch.tensor): messages, size N*M
        samples (int, optional): default 5000 - number of pairs to sample
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in representation/messages
    """
    dataset_length = compositional_representation.shape[0]
    generated_sequences = one_hot(generated_sequences.numpy()).reshape(
        dataset_length, -1
    )

    sim_representation = np.zeros(samples)
    sim_sequences = np.zeros(samples)

    for i in range(samples):
        rnd = np.random.choice(dataset_length, 2, replace=False)
        s1, s2 = rnd[0], rnd[1]

        sim_representation[i] = scipy.spatial.distance.hamming(
            compositional_representation[s1], compositional_representation[s2]
        )

        sim_sequences[i] = scipy.spatial.distance.cosine(
            generated_sequences[s1], generated_sequences[s2]
        )

    return scipy.stats.pearsonr(sim_sequences, sim_representation)[0]


def get_topographical_similarity(trainer, dataloader):
    sequences = []
    metas = []
    with torch.no_grad():
        for (batch, targets) in dataloader:
            loss, acc, seq = trainer(batch, targets)
            metas.append(batch)
            sequences.append(seq)

    sequences = torch.cat(sequences, 0)
    metas = torch.cat(metas, 0)
    return calc_topographical_similarity(metas, sequences)
