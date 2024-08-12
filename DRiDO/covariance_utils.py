import numpy as np
import pandas as pd


def get_near_psd(A, tol=1e-5):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eigh(C)
    eigval[eigval < tol] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def make_npn_corr(df, ensure_spsd=True, diag_exaggeration=1e-1):
    """Compute nonparanormal SKEPTIC corr estimator

    Reference: https://arxiv.org/pdf/1206.6488.pdf

    Seems to be a typo in the actual paper (extra factor of 2).
    """
    npn_cov = np.sin(np.pi / 2 * df.corr(method='kendall'))
    variables = npn_cov.columns
    npn_cov = npn_cov.values
    np.fill_diagonal(npn_cov, 1 / diag_exaggeration)
    npn_cov *= diag_exaggeration
    if ensure_spsd:
        npn_cov = get_near_psd(npn_cov)
    return pd.DataFrame(npn_cov, index=variables, columns=variables)


def normalize_diag(arr, clip_to_eps=True):
    """Normalize each X_ij by sqrt(X_ii * X_jj) (e.g., when converting cov
    to correlation or precision to partial correlation"""
    denom = np.sqrt(np.diag(arr)[None, :] * np.diag(arr)[:, None])
    if clip_to_eps:
        denom = np.clip(denom, 1e-9, np.inf)
    return arr / denom


def remove_diag(arr):
    """Subtract the diagonal which is useful for plotting."""
    return arr - np.diag(np.diag(arr))


def compute_partial_correlation(precision, should_remove_diag=True):
    # Ignore sign change on the diagonal for now since we add it back later
    partial_correlation = -normalize_diag(precision)
    partial_correlation = remove_diag(partial_correlation)
    if not should_remove_diag:
        # Self-correlation is always 1
        partial_correlation += np.eye(partial_correlation.shape[0])
    return partial_correlation
