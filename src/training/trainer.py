"""
Generic training loop for Hurst exponent estimation models.
Handles training, validation, early stopping, and LR scheduling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import time


class HurstTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        batch_size: int = 256,
        patience: int = 10,
        save_dir: str = "models/saved",
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
    ) -> dict:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"Training {self.model_name} on {self.device}...")
        print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    pred = self.model(X_batch)
                    loss = self.criterion(pred, y_batch)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir / f"{self.model_name}_best.pt")
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | {elapsed:.0f}s")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

        # Load best model
        self.model.load_state_dict(torch.load(self.save_dir / f"{self.model_name}_best.pt", weights_only=True))
        total_time = time.time() - start_time
        print(f"  Training complete in {total_time:.1f}s (best epoch: {best_epoch})")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())
        return np.concatenate(predictions)

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout: keep dropout on, run multiple forward passes."""
        self.model.train()  # keeps dropout active
        all_preds = []
        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)

        for _ in range(n_samples):
            preds = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    X_batch = X_batch.to(self.device)
                    pred = self.model(X_batch)
                    preds.append(pred.cpu().numpy())
            all_preds.append(np.concatenate(preds))

        all_preds = np.stack(all_preds)  # (n_samples, n_data)
        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)

        self.model.eval()
        return mean, std
