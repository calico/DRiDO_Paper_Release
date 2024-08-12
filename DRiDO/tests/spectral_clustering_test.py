import networkx as nx
import numpy as np
import unittest
import jax_cr.spectral_clustering as sc


class Test_affinityToLaplacianEigenvectors(unittest.TestCase):
    def test_affinityToLaplacianSpectra(self):
        n = 10
        cycle_graph = nx.cycle_graph(n)
        affinity_matrix = nx.to_numpy_array(cycle_graph)
        result, _ = sc.affinity_to_laplacian_eigenvectors(affinity_matrix)
        # Known specta for ring of length n
        # Reference: https://www.johndcook.com/blog/2016/01/09/spectra-of-complete-graphs-stars-and-rings/
        analytic_spectra = sorted(np.array(
            [2 - 2 * np.cos((2 * np.pi * k) / n) for k in range(n)]))
        self.assertTrue(np.allclose(analytic_spectra, result))


class Test_givensRotations(unittest.TestCase):
    def test_givensRotations90deg(self):
        n = 5
        theta = np.pi / 2
        basis = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rot = sc.generate_Givens_rotation(i, j, theta, n)
                self.assertTrue(np.allclose(
                    rot.dot(basis[i]), basis[j], atol=1e-7))

    def test_givensRotations45deg(self):
        n = 5
        theta = np.pi / 4
        basis = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rot = sc.generate_Givens_rotation(i, j, theta, n)
                analytic = (basis[i] + basis[j]) * (1 / np.sqrt(2))
                self.assertTrue(np.allclose(
                    rot.dot(basis[i]), analytic, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
