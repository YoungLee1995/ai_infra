import tempfile
import unittest
from pathlib import Path

import torch

from ddp_baseline.train import MLP, SyntheticBinaryDataset, build_scheduler, load_checkpoint, save_checkpoint


class TrainComponentTests(unittest.TestCase):
    def test_dataset_is_reproducible(self):
        first = SyntheticBinaryDataset(samples=12, seed=11)
        second = SyntheticBinaryDataset(samples=12, seed=11)
        self.assertTrue(torch.equal(first.features, second.features))
        self.assertTrue(torch.equal(first.labels, second.labels))

    def test_checkpoint_restores_model_and_optimizer(self):
        model = MLP()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = build_scheduler(optimizer, total_steps=20, warmup_steps=2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loader_generator = torch.Generator().manual_seed(9)
        rng_states = [{"python": __import__("random").getstate(), "torch": torch.get_rng_state(), "loader": loader_generator.get_state()}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            config = {
                "device": "cpu", "samples": 100, "input_dim": 32, "hidden_dim": 64,
                "batch_size": 8, "accumulation_steps": 1, "learning_rate": 0.01,
                "seed": 1, "amp": False, "warmup_steps": 2, "world_size": 1,
            }
            save_checkpoint(path, model, optimizer, scaler, 2, 5, 13, 40, scheduler, rng_states, config)
            restored_model = MLP()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            restored_scheduler = build_scheduler(restored_optimizer, total_steps=20, warmup_steps=2)
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            restored_loader_generator = torch.Generator().manual_seed(1)
            position = load_checkpoint(
                path, restored_model, restored_optimizer, restored_scaler, restored_scheduler,
                torch.device("cpu"), restored_loader_generator, config,
            )
            self.assertEqual(position, {"epoch": 2, "batch_in_epoch": 5, "global_step": 13, "samples_seen": 40})
            for expected, actual in zip(model.parameters(), restored_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
