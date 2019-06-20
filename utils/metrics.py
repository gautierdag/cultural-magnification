import torch
import itertools
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.metrics import jaccard_similarity_score


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
    generated_sequences = one_hot(generated_sequences.cpu().numpy()).reshape(
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


def message_distance(messages1, messages2):
    """
    Args:
        message: N messages of length L from A agents, shape: N*A*L

    """
    if hasattr(messages1, "numpy"):
        messages1 = messages1.cpu().numpy()
    if hasattr(messages2, "numpy"):
        messages2 = messages2.cpu().numpy()
    messages = np.stack([messages1, messages2], axis=1)

    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    encoded_messages = one_hot(messages).reshape(N, A, -1).astype(float)
    tot_dist = 0
    perfect_matches = 0
    for c in combinations:
        diff = np.sum(
            np.abs(encoded_messages[:, c[0], :] - encoded_messages[:, c[1], :]), axis=1
        )
        perfect_matches += np.count_nonzero(diff == 0)
        tot_dist += np.sum(diff)

    # average over number of number of combinations and examples
    tot_dist /= N * len(combinations)
    perfect_matches /= N * len(combinations)

    return tot_dist, perfect_matches


def jaccard_similarity(messages1, messages2, samples=1000):
    """
    Averages average jaccard similarity between all pairs of agents.
    Args:
        messages (ndarray, ints): N messages of length L from A agents, shape: N*A*L
    Returns:
        score (float): average jaccard similarity between all pairs of agents.
    """
    if hasattr(messages1, "numpy"):
        messages1 = messages1.cpu().numpy()
    if hasattr(messages2, "numpy"):
        messages2 = messages2.cpu().numpy()
    messages = np.stack([messages1, messages2], axis=1)

    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    score = 0.0
    for c in combinations:
        for _ in range(samples):
            s = np.random.randint(N)
            score += jaccard_similarity_score(
                messages[s, c[0], :], messages[s, c[1], :]
            )

    # average over number of combinations
    score /= len(combinations) * samples

    return score


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
