"""Uploader tests: stdlib only, no downloads, no credential use, no large files."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from publish_hf import strip_remote_paths, validate_checkpoint, verify


class UploadTests(unittest.TestCase):
    def test_paths_removed_without_mutating_original(self):
        original = {"sae_path": "/server/checkpoint", "tokens": 100,
                    "checkpoints": [{"path": "/server/ckpt", "reload_check": {"passed": True}}]}
        snapshot = copy.deepcopy(original)
        cleaned = strip_remote_paths(original)
        self.assertEqual(original, snapshot)
        self.assertNotIn("sae_path", cleaned)
        self.assertEqual(cleaned["checkpoints"], [{"reload_check": {"passed": True}}])

    def fixture(self):
        records = [{"path": "sae/weights", "bytes": 100, "sha256": "abc"},
                   {"path": "sae/config", "bytes": 3, "sha256": "def", "git_blob_sha1": "123"}]
        entries = [SimpleNamespace(path="sae/weights", size=100, lfs=SimpleNamespace(sha256="abc"), blob_id="unused"),
                   SimpleNamespace(path="sae/config", size=3, lfs=None, blob_id="123")]
        api = SimpleNamespace(get_paths_info=lambda *args, **kwargs: entries)
        return api, records, entries

    def test_verify_lfs_and_git_hashes(self):
        api, records, _ = self.fixture()
        self.assertTrue(verify(api, records, "pinned-commit")["passed"])

    def test_reject_missing_or_changed_files(self):
        for mutation in ("missing", "lfs_hash", "size", "git_hash"):
            api, records, entries = self.fixture()
            if mutation == "missing":
                entries.pop()
            elif mutation == "lfs_hash":
                entries[0].lfs.sha256 = "wrong"
            elif mutation == "size":
                entries[0].size = 99
            else:
                entries[1].blob_id = "wrong"
            with self.assertRaises(ValueError):
                verify(api, records, "pinned-commit")

    def test_intermediate_checkpoint_identity(self):
        cfg = {"architecture": "topk", "d_in": 4096, "d_sae": 32768, "k": 50,
               "metadata": {"training_tokens": 10_000_000, "hook_name": "blocks.8.ln1.hook_normalized",
                            "model_name": "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A",
                            "model_revision": "027f599bb4c24e4bac72932ce557f9fa325aa9be"}}
        checkpoint = {"tokens": 10_000_000, "reload_check": {"passed": True}}
        for mutation in (None, "tokens", "hook", "reload"):
            bad_cfg, bad_checkpoint = copy.deepcopy(cfg), copy.deepcopy(checkpoint)
            if mutation == "tokens":
                bad_cfg["metadata"]["training_tokens"] = 100_000_000
            elif mutation == "hook":
                bad_cfg["metadata"]["hook_name"] = "blocks.8.hook_attn_out"
            elif mutation == "reload":
                bad_checkpoint["reload_check"]["passed"] = False
            with patch("publish_hf.small_json", side_effect=[bad_cfg, bad_checkpoint]), \
                 patch.object(Path, "resolve", autospec=True, side_effect=lambda p, **kw: p), \
                 patch.object(Path, "is_symlink", return_value=False), \
                 patch.object(Path, "is_file", return_value=True), \
                 patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1_073_889_608)):
                if mutation:
                    with self.assertRaises(ValueError):
                        validate_checkpoint(Path("/remote/exact-run"), 8, 10_000_000)
                else:
                    folder, _, _ = validate_checkpoint(Path("/remote/exact-run"), 8, 10_000_000)
                    self.assertEqual(folder.name, "tokens_010000000")


if __name__ == "__main__":
    unittest.main()
