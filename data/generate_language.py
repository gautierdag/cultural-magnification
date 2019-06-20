import numpy as np
from .AgentVocab import AgentVocab


def generate_uniform_language_fixed_length(vocabulary_object, count: int, max_length: int):
    """
    Generates a language sampled uniformly from the vocabulary in the vocabulary object
    """
    sentences = np.random.randint(1, vocabulary_object.vocab_size+1, size=(count, max_length+1))
    sentences[:,-1] = vocabulary_object.eos  # add eos after every sentence
    return sentences.astype(np.int64)

def generate_uniform_language_varying_length(vocabulary_object, count: int, max_length: int):
    """
    Varies the length of the sentences
    """
    sentences = np.random.randint(1, vocabulary_object.vocab_size+1, size=(count, max_length+1))
    lengths = np.random.randint(1, max_length+1, size=count)  # sample lengths uniformly (minimum length of 1)
    mask = np.zeros((count, max_length+1))
    for i in range(count):
        mask[i, :lengths[i]] = 1
    sentences = sentences*mask
    sentences[np.arange(count), lengths] = vocabulary_object.eos # append eos after each sentence

    return sentences.astype(np.int64)