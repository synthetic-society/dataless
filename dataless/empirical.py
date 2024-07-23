import pandas as pd
import numpy as np
import scipy.stats as sstats


def frequencies(arr : np.ndarray) -> np.ndarray:
    """
    Return the sorted frequencies of a numpy 1D array.
    """
    freqs = np.unique(arr, return_counts=True)[1]
    return np.sort(freqs)


def empirical_entropy(arr : np.ndarray, base=np.e) -> float:
    """
    Return the empirical entropy of a numpy 1D array,
    according to a given base (default: e).
    """
    freqs = frequencies(arr)
    return sstats.entropy(freqs / len(arr), base=base)


def counts_from_dataframe(df : pd.DataFrame) -> np.ndarray:
    """
    Return the sorted frequencies from a pandas Dataframe
    (the frequencies of unique rows).
    """
    arr = df.values
    shape_row = arr.shape[1]
    rowtype = np.dtype((np.void, arr.dtype.itemsize * shape_row))
    arr_flat = np.ascontiguousarray(arr).view(rowtype)
    arr_flat = arr_flat.reshape(-1, arr_flat.shape[1])
    return frequencies(arr_flat)


def uniqueness(freqs : np.ndarray) -> float:
    """ Return the average uniqueness from a vector of frequencies. """
    return (freqs == 1).sum() / freqs.sum()


def correctness(freqs : np.ndarray) -> float:
    """ Return the average correctness from a vector of frequencies. """
    return len(freqs) / freqs.sum()
