import numpy as np
from numpy.random import default_rng
import pandas as pd
import unittest

import jax_cr.covariance_utils


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


class Test_getNearPSD(unittest.TestCase):
    def test_nearpsd(self):
        eps = 1e-10
        x = np.random.randn(10, 10)
        psd_x = jax_cr.covariance_utils.get_near_psd(x)
        eigenvals, _ = np.linalg.eigh(psd_x)
        self.assertTrue(np.allclose(psd_x - psd_x.T, 0))
        self.assertTrue(np.all(eigenvals > -eps))
        psd_x2 = jax_cr.covariance_utils.get_near_psd(psd_x)
        self.assertTrue(np.allclose(psd_x, psd_x2))


class Test_normalizeDiag(unittest.TestCase):
    def test_normalize_diag(self):
        x = np.random.randn(10, 10)
        np.fill_diagonal(x, np.abs(np.diag(x)))
        test_x = jax_cr.covariance_utils.normalize_diag(x)
        self.assertTrue(np.allclose(np.diag(test_x), 1))
        for i in range(10):
            for j in range(10):
                self.assertTrue(test_x[i, j] == x[i, j] /
                                np.sqrt(x[i, i] * x[j, j]))


class Test_makeNPNCov(unittest.TestCase):
    def test_makeNPNCorr(self):
        # Estimator should be close to exact for N-d Gaussian
        dummy_samples = make_dummy_data()
        emp_cov = np.cov(dummy_samples)
        emp_corr = jax_cr.covariance_utils.normalize_diag(emp_cov)
        dummy_samples_df = pd.DataFrame(
            dummy_samples.T, columns=['a', 'b', 'c', 'd'])
        npn_estimator = jax_cr.covariance_utils.make_npn_corr(
            dummy_samples_df, ensure_spsd=False, diag_exaggeration=1)
        self.assertTrue(np.allclose(emp_corr, npn_estimator.values, atol=1e-2))

    def test_makeNPNCorr_log(self):
        # Test on log-normal
        dummy_samples = make_dummy_data()
        emp_cov = np.cov(dummy_samples)
        emp_corr = jax_cr.covariance_utils.normalize_diag(emp_cov)
        log_norm_samples = np.exp(dummy_samples)
        log_norm_samples_df = pd.DataFrame(
            log_norm_samples.T, columns=['a', 'b', 'c', 'd'])
        npn_estimator = jax_cr.covariance_utils.make_npn_corr(
            log_norm_samples_df, ensure_spsd=False, diag_exaggeration=1)
        self.assertTrue(np.allclose(emp_corr, npn_estimator.values, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
