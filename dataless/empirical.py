import numpy as np
import pandas as pd
import scipy.stats as sstats


def frequencies(arr: np.ndarray) -> np.ndarray:
    """Calculate the sorted frequencies of unique elements in an array.

    Args:
        arr: 1D numpy array of observations

    Returns:
        ndarray: Sorted array of frequencies

    """
    freqs = np.unique(arr, return_counts=True)[1]
    return np.sort(freqs)


def empirical_entropy(arr: np.ndarray, base=np.e) -> float:
    """Calculate the empirical (biased) entropy of a sample.

    Args:
        arr: 1D numpy array of observations
        base: Base for the logarithm (default: e for natural logarithm)

    Returns:
        float: Empirical entropy value

    Raises:
        ValueError: If input array is empty

    """
    if len(arr) == 0:
        raise ValueError("Cannot compute entropy of empty array")
    freqs = frequencies(arr)
    return sstats.entropy(freqs / len(arr), base=base)


def counts_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Calculate frequencies of unique rows in a DataFrame.

    The function has been optimized for speed by converting the DataFrame
    to a contiguous array of bytes and then counting the frequencies of
    unique rows. This is much faster than counting the frequencies of
    individual rows in the DataFrame.

    Args:
        df: pandas DataFrame where each row is an observation

    Returns:
        ndarray: Sorted array of frequencies for unique rows

    """
    arr = df.values
    shape_row = arr.shape[1]
    rowtype = np.dtype((np.void, arr.dtype.itemsize * shape_row))
    arr_flat = np.ascontiguousarray(arr).view(rowtype)
    arr_flat = arr_flat.reshape(-1, arr_flat.shape[1])
    return frequencies(arr_flat)


def uniqueness(freqs: np.ndarray) -> float:
    """Calculate the empirical uniqueness from frequency counts.

    Uniqueness is defined as the proportion of observations that appear
    exactly once in the sample.

    Args:
        freqs: Array of frequency counts

    Returns:
        float: Empirical uniqueness value in [0,1]

    Raises:
        ValueError: If input array is empty

    """
    if len(freqs) == 0:
        raise ValueError("Cannot compute uniqueness of empty array")
    total = freqs.sum()
    if total == 0:
        raise ValueError("Total frequency count cannot be zero")
    return (freqs == 1).sum() / total


def correctness(freqs: np.ndarray) -> float:
    """Calculate the empirical correctness from frequency counts.

    Correctness is calculated as the ratio of unique frequencies to total observations.

    Args:
        freqs: Array of frequency counts

    Returns:
        float: Empirical correctness value in [0,1]

    Raises:
        ValueError: If input array is empty

    """
    if len(freqs) == 0:
        raise ValueError("Cannot compute correctness of empty array")
    total = freqs.sum()
    if total == 0:
        raise ValueError("Total frequency count cannot be zero")
    return len(freqs) / total
