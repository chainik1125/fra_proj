import itertools
import unittest

import torch

from experiment import Reader, ShamirHMM


class ShamirChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        f = torch.eye(50, 100)
        self.geometry = {"features": f, "decoder": f}

    def test_exact_sharing_and_single_share_secrecy(self):
        p = 5
        counts = torch.zeros(2, p, p, dtype=torch.long)
        for secret, slope in itertools.product(range(p), repeat=2):
            y1, y2 = (secret + slope) % p, (secret + 2 * slope) % p
            self.assertEqual((2 * y1 - y2) % p, secret)
            counts[0, y1, secret] += 1
            counts[1, y2, secret] += 1
        torch.testing.assert_close(counts, torch.ones_like(counts))

    def test_oracle_against_enumerated_polynomials(self):
        data = ShamirHMM(self.geometry)
        batch = data.sample(100, torch.Generator().manual_seed(4))
        for i in range(100):
            q = batch["request"][i]
            shares = []
            for t in range(6):
                if batch["valid"][i, t] and batch["z"][i, t, q] > 0:
                    atom = int(batch["z"][i, t, 3:13].argmax())
                    shares.append((atom // 5 + 1, atom % 5))
            possible = [
                s
                for s, a in itertools.product(range(5), repeat=2)
                if all((s + a * x) % 5 == y for x, y in shares)
            ]
            expected = data.secret_f[possible].mean(0)
            torch.testing.assert_close(batch["oracle"][i], expected)

    def test_positional_output_is_query_blind(self):
        data = ShamirHMM(self.geometry)
        batch = data.sample(10, torch.Generator().manual_seed(2), complete=True)
        for readout in ["linear", "mlp"]:
            model = Reader(routing="position", readout=readout)
            original = model(batch)
            batch["query"] = torch.randn_like(batch["query"]) * 10
            torch.testing.assert_close(original, model(batch))

    def test_query_blind_mse_bound(self):
        data = ShamirHMM(self.geometry)
        secrets = torch.tensor(list(itertools.product(range(5), repeat=3)))
        target = data.secret_f[secrets]
        optimal = target.mean(1, keepdim=True)
        mse = (target - optimal).square().sum(-1).mean() / data.variance
        self.assertAlmostEqual(mse.item(), 2 / 3, places=6)

    def test_exact_shares_and_fra_scores(self):
        data = ShamirHMM(self.geometry)
        batch = data.sample(100, torch.Generator().manual_seed(72), complete=True)
        torch.testing.assert_close(
            (2 * batch["requested_shares"][:, 0] - batch["requested_shares"][:, 1]) % 5,
            batch["target"],
        )
        self.assertTrue(torch.all(batch["count"] == 2))
        model = Reader()
        _, a, pooled, scores = model(batch, details=True)
        b = data.f @ model.wq @ model.wk.T @ data.f.T / 10
        expected = torch.einsum("bi,ij,btj->bt", batch["qz"], b, batch["z"])
        torch.testing.assert_close(scores, expected)
        ov = data.f @ model.wv @ model.wo
        expected_pool = torch.einsum("bt,bti,id->bd", a, batch["z"], ov) + model.bias
        torch.testing.assert_close(pooled, expected_pool)


if __name__ == "__main__":
    unittest.main()
