import torch
import itertools
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.metrics import jaccard_score
import warnings


def one_hot(a, ncols):
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def calc_topographical_similarity(
    compositional_representation, generated_sequences, vocab_size
):
    """
    Calculates Topological Similarity using all possible pair combinations
    Args:
        compositional_representation (np.array): one-hot encoded compositional, size N*C
        messages (torch.tensor): messages, size N*M
        vocab_size: vocab size to encode in one hot representation
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in representation/messages
    """
    dataset_length = compositional_representation.shape[0]
    generated_sequences = one_hot(
        generated_sequences.cpu().numpy(), vocab_size
    ).reshape(dataset_length, -1)

    combinations = list(itertools.combinations(range(dataset_length), 2))
    sim_representation = np.zeros(len(combinations))
    sim_sequences = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]
        sim_representation[i] = scipy.spatial.distance.hamming(
            compositional_representation[s1], compositional_representation[s2]
        )
        sim_sequences[i] = scipy.spatial.distance.hamming(
            generated_sequences[s1], generated_sequences[s2]
        )
    # check if standard deviation is not 0
    if sim_sequences.std() == 0.0 or sim_representation.std() == 0.0:
        warnings.warn(
            "Standard deviation of 0.0 for passed parameter in compositionality_metrics"
        )
        topographic_similarity = 0
    else:
        topographic_similarity = scipy.stats.pearsonr(
            sim_sequences, sim_representation
        )[0]

    return topographic_similarity


def calc_jaccard_topographical_similarity(
    compositional_representation, generated_sequences, average="macro"
):
    """
    Calculates Topological Similarity using all possible pair combinations
    Args:
        compositional_representation (np.array): one-hot encoded compositional, size N*C
        messages (torch.tensor): messages, size N*M
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in representation/messages
    """
    dataset_length = compositional_representation.shape[0]

    combinations = list(itertools.combinations(range(dataset_length), 2))

    if hasattr(generated_sequences, "numpy"):
        generated_sequences = generated_sequences.cpu().numpy()

    sim_representation = np.zeros(len(combinations))
    sim_sequences = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]
        sim_representation[i] = scipy.spatial.distance.hamming(
            compositional_representation[s1], compositional_representation[s2]
        )
        sim_sequences[i] = 1 - jaccard_score(
            generated_sequences[s1], generated_sequences[s2], average=average
        )

    # check if standard deviation is not 0
    if sim_sequences.std() == 0.0 or sim_representation.std() == 0.0:
        warnings.warn(
            "Standard deviation of 0.0 for passed parameter in compositionality_metrics"
        )
        topographic_similarity = 0
    else:
        topographic_similarity = scipy.stats.pearsonr(
            sim_sequences, sim_representation
        )[0]

    return topographic_similarity


def message_distance(messages1, messages2, vocab_size):
    """
    Args:
        message: N messages of length L from two separate languages
        vocab_size: vocab size to encode in one hot representation
    """
    if hasattr(messages1, "numpy"):
        messages1 = messages1.cpu().numpy()
    if hasattr(messages2, "numpy"):
        messages2 = messages2.cpu().numpy()

    N, L = messages1.shape
    assert N == messages2.shape[0]

    encoded_messages1 = one_hot(messages1, vocab_size).astype(float)
    encoded_messages2 = one_hot(messages2, vocab_size).astype(float)

    diff = np.sum(np.abs(encoded_messages1 - encoded_messages2).reshape(N, -1), axis=1)
    perfect_matches = np.count_nonzero(diff == 0)
    tot_dist = np.sum(diff)

    # normalize over number of number examples (also by the maximum total distance possible)
    tot_dist /= N * (L * 2)
    perfect_matches /= N

    return tot_dist, perfect_matches


def jaccard_similarity(messages1, messages2, average="macro"):
    """
    Averages average jaccard similarity between two agents.
    Args:
        messages (ndarray, ints): N messages of length L two different agents
        average {'macro', 'micro'}:     
            'micro': Calculate metrics globally by counting the total true positives, 
                    false negatives and false positives.
            'macro': Calculate metrics for each label, and find their unweighted mean. 
                    This does not take label imbalance into account. 
    Returns:
        score (float): average jaccard similarity between agents.
    """
    if hasattr(messages1, "numpy"):
        messages1 = messages1.cpu().numpy()
    if hasattr(messages2, "numpy"):
        messages2 = messages2.cpu().numpy()

    N, L = messages1.shape
    assert N == messages2.shape[0]

    score = 0.0
    for i in range(N):
        score += jaccard_score(messages1[i], messages2[i], average=average)

    # normalize over number of combinations
    score /= N

    return score


def custom_topo(compositional_representation, generated_sequences):
    """
    Calculates Topological Similarity using all possible pair combinations
    Args:
        compositional_representation (np.array): one-hot encoded compositional, size N*C
        messages (torch.tensor): messages, size N*M
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in representation/messages
    """
    dataset_length = compositional_representation.shape[0]

    combinations = list(itertools.combinations(range(dataset_length), 2))

    if hasattr(generated_sequences, "numpy"):
        generated_sequences = generated_sequences.cpu().numpy()

    sim_representation = np.zeros(len(combinations))
    sim_sequences = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]
        sim_representation[i] = scipy.spatial.distance.hamming(
            compositional_representation[s1], compositional_representation[s2]
        )
        iou = len(
            set(generated_sequences[s1]).intersection(set(generated_sequences[s2]))
        ) / len(set(generated_sequences[s1]).union(set(generated_sequences[s2])))

        sim_sequences[i] = 1.0 - iou

    # check if standard deviation is not 0
    if sim_sequences.std() == 0.0 or sim_representation.std() == 0.0:
        warnings.warn("Standard deviation of 0.0 for passed parameter in custom_topo")
        topographic_similarity = 0
    else:
        topographic_similarity = scipy.stats.pearsonr(
            sim_sequences, sim_representation
        )[0]

    return topographic_similarity
