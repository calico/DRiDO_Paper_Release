
import itertools
import numpy as np
from numpy.random import default_rng
import unittest

from jax_cr.covariance_utils import compute_partial_correlation, normalize_diag
from jax_cr.path_decomposition import compute_all_paths_scores


def make_dummy_data():
    # Graphical model
    #
    #       A
    #      / \
    #     /   \
    #    /     \
    #   B-------C
    #    \     /
    #     \   /
    #      \ /
    #       D
    #
    rng = default_rng(seed=0)
    n = int(1e4)
    a = rng.normal(scale=1, size=n)
    b = a + rng.normal(scale=1, size=n)
    c = a + 2*b + rng.normal(scale=1, size=n)
    d = b + c + rng.normal(scale=1, size=n)
    samples = np.stack([a, b, c, d])
    return samples


class Test_pathDecomposition(unittest.TestCase):
    def test_compute_all_paths_scores(self):
        samples = make_dummy_data()
        cov = np.cov(samples)
        corr = normalize_diag(cov)
        precision = np.linalg.inv(cov)
        pcorr = compute_partial_correlation(
            precision, should_remove_diag=False)
        # Path scores should sum to the marginal correlations
        for i, j in itertools.combinations(range(4), 2):
            path_scores_df = compute_all_paths_scores(pcorr, i, j)
            self.assertTrue(np.isclose(
                corr[i, j], path_scores_df.sum()['score']))


if __name__ == '__main__':
    unittest.main()