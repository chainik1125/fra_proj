"""Independent checks of the generative law, oracle and FRA conventions."""

import itertools
from pathlib import Path
import sys
import unittest

import torch

from attention import Process, SingleHead


class SyntheticChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.f = torch.linalg.qr(torch.randn(7, 3)).Q.T
        self.geometry = {
            "features": self.f,
            "probs": torch.tensor([0.2, 0.4, 0.7]),
            "correlation": torch.eye(3),
            "decoder": self.f,
        }

    def test_fra_score_and_ov_exactness(self):
        model = SingleHead(7, 5)
        z = torch.rand(2, 5, 3)
        x = z @ self.f
        pred, a, scores = model(x, return_details=True)
        b = self.f @ model.wq @ model.wk.T @ self.f.T / 7**0.5
        ov = self.f @ model.wv @ model.wo
        torch.testing.assert_close(
            z @ b @ z.transpose(-1, -2), scores, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            (a @ z) @ ov + model.bias, pred, atol=1e-6, rtol=1e-5
        )

    def test_repository_fra_implementation(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from fra.core.fra import compute_fra_sparse

        model = SingleHead(7, 5)
        z = torch.rand(5, 3)
        z[z < 0.5] = 0
        fra = compute_fra_sparse(z, self.f, model.wq, model.wk, 7**0.5)
        b = self.f @ model.wq @ model.wk.T @ self.f.T / 7**0.5
        expected = torch.einsum("ti,ij,sj->tsij", z, b, z)
        expected = expected.masked_fill(model.future[:, :, None, None], 0)
        torch.testing.assert_close(fra.to_dense(), expected, atol=1e-6, rtol=1e-5)

    def test_clean_oracle_and_reset_stationarity(self):
        process = Process(self.geometry, "markov", rho=0.7, magnitude_std=0.0)
        _, z = process.sample(20000, 6, torch.Generator().manual_seed(19))
        expected = (0.3 * process.pi + 0.7 * z) @ self.f
        torch.testing.assert_close(process.conditional_mean(z), expected)
        torch.testing.assert_close(z.mean((0, 1)), process.pi, atol=0.009, rtol=0.0)
        covariance = ((z[:, :-1] - process.pi) * (z[:, 1:] - process.pi)).mean((0, 1))
        torch.testing.assert_close(
            covariance / (process.pi * (1 - process.pi)),
            torch.full((3,), 0.7),
            atol=0.012,
            rtol=0.0,
        )

    def test_noisy_filter_against_enumeration(self):
        process = Process(
            self.geometry, "markov", rho=0.7, hit=0.625, magnitude_std=0.0
        )
        observations = torch.tensor(
            [[[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]
        )
        actual = process.conditional_mean(observations) @ self.f.T
        for feature in range(3):
            pi = process.pi[feature].item()
            for t in range(3):
                numerator = denominator = 0.0
                for states in itertools.product([0, 1], repeat=t + 1):
                    prob = pi if states[0] else 1 - pi
                    for k in range(t + 1):
                        if k:
                            transition = 0.3 * pi + 0.7 * states[k - 1]
                            prob *= transition if states[k] else 1 - transition
                        emit = 0.625 * states[k]
                        prob *= emit if observations[0, k, feature] else 1 - emit
                    numerator += prob * (0.3 * pi + 0.7 * states[-1]) * 0.625
                    denominator += prob
                self.assertAlmostEqual(
                    actual[0, t, feature].item(), numerator / denominator, places=6
                )

    def test_independence_does_not_imply_zero_raw_cross_products(self):
        process = Process(self.geometry, "independent_iid", magnitude_std=0.0)
        _, z = process.sample(50000, 2, torch.Generator().manual_seed(31))
        raw = z[:, 0].T @ z[:, 1] / len(z)
        torch.testing.assert_close(
            raw, process.pi[:, None] * process.pi[None, :], atol=0.007, rtol=0.0
        )

    def test_causal_mask(self):
        model = SingleHead(7, 5)
        x = torch.randn(2, 5, 7)
        y = model(x)
        changed = x.clone()
        changed[:, 3:] += 100
        torch.testing.assert_close(y[:, :3], model(changed)[:, :3])

    def test_offdiagonal_background_selects_exact_binary_pattern(self):
        n = 6
        patterns = torch.tensor(list(itertools.product([0.0, 1.0], repeat=n)))
        queries = patterns[1:]
        b = (n + 1) * torch.eye(n) - torch.ones(n, n)
        scores = queries @ b @ patterns.T
        torch.testing.assert_close(patterns[scores.argmax(-1)], queries)
        sorted_scores = scores.sort(-1).values
        self.assertTrue(torch.all(sorted_scores[:, -1] > sorted_scores[:, -2]))
        diagonal_scores = queries @ torch.eye(n) @ patterns.T
        # Every non-full nonempty query ties with at least one strict superset.
        num_ties = (diagonal_scores == diagonal_scores.max(-1).values[:, None]).sum(-1)
        self.assertTrue(torch.all(num_ties[:-1] > 1))


if __name__ == "__main__":
    unittest.main()
