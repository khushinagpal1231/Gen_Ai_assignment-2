"""Lab 2 - Image Synthesis Using GANs (Fashion-MNIST).

ETMMGA304 Laboratory Manual, pages 7-9.

Trains an MLP generator/discriminator pair with alternating
BCEWithLogitsLoss updates and saves:

    lab_outputs/gan_epoch_01.png, gan_epoch_05.png, gan_epoch_10.png
    lab_outputs/gan_losses.png
    lab_outputs/fashion_generator.pt

Run:
    python lab2_gan_fashion_mnist.py [--epochs 10] [--batch-size 128]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

SEED = 42
OUT_DIR = Path("lab_outputs")
DATA_DIR = Path("data")
LATENT_DIM = 64
GRID_EPOCHS = (1, 5, 10)


def setup(seed: int = SEED) -> torch.device:
    """Seed RNGs, create output folders and return the compute device."""
    random.seed(seed)
    torch.manual_seed(seed)
    OUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")
    return device


# --------------------------------------------------------------------------- #
# Step 1 - Data scaled to [-1, 1] to match the generator's tanh output
# --------------------------------------------------------------------------- #
def load_fashion_mnist() -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return datasets.FashionMNIST(str(DATA_DIR), train=True, download=True,
                                 transform=transform)


# --------------------------------------------------------------------------- #
# Step 2 - Networks
# --------------------------------------------------------------------------- #
def build_generator(latent_dim: int = LATENT_DIM) -> nn.Sequential:
    """64 -> 256 -> 512 -> 784, LeakyReLU hidden layers, Tanh output."""
    return nn.Sequential(
        nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 512), nn.LeakyReLU(0.2),
        nn.Linear(512, 784), nn.Tanh(),
    )


def build_discriminator() -> nn.Sequential:
    """784 -> 512 -> 256 -> 1 logit (no sigmoid; BCEWithLogitsLoss)."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.LeakyReLU(0.2),
        nn.Linear(512, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
    )


class GANTrainer:
    """Holds both networks, their optimizers and the fixed grid noise."""

    def __init__(self, device: torch.device, lr: float = 0.0002,
                 betas: tuple[float, float] = (0.5, 0.999)) -> None:
        self.device = device
        self.G = build_generator().to(device)
        self.D = build_discriminator().to(device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        self.fixed_z = torch.randn(25, LATENT_DIM, device=device)

    # ------------------------------------------------------------------- #
    # Step 3 - Alternating updates
    # ------------------------------------------------------------------- #
    def train_batch(self, real: torch.Tensor) -> tuple[float, float]:
        n = real.size(0)
        fake = self.G(torch.randn(n, LATENT_DIM, device=self.device))

        # Discriminator: real -> 1, detached fake -> 0.
        real_logits = self.D(real)
        fake_logits = self.D(fake.detach())
        d_loss = (
            self.loss_fn(real_logits, torch.ones_like(real_logits))
            + self.loss_fn(fake_logits, torch.zeros_like(fake_logits))
        ) / 2
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()

        # Generator: fool the (frozen) discriminator into predicting 1.
        self.D.requires_grad_(False)
        self.opt_g.zero_grad()
        logits = self.D(fake)
        g_loss = self.loss_fn(logits, torch.ones_like(logits))
        g_loss.backward()
        self.opt_g.step()
        self.D.requires_grad_(True)
        return d_loss.item(), g_loss.item()

    @torch.no_grad()
    def generate_grid(self, epoch: int) -> Path:
        """Render the 25 fixed-noise samples as a 5x5 grid."""
        self.G.eval()
        images = self.G(self.fixed_z).cpu().reshape(-1, 1, 28, 28)
        self.G.train()
        grid = make_grid((images + 1) / 2, nrow=5)
        path = OUT_DIR / f"gan_epoch_{epoch:02d}.png"
        plt.figure(figsize=(5, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Epoch {epoch}")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")
        return path


# --------------------------------------------------------------------------- #
# Step 4 - Training loop and evaluation artefacts
# --------------------------------------------------------------------------- #
def train(trainer: GANTrainer, loader: DataLoader,
          epochs: int = 10) -> list[tuple[float, float]]:
    history: list[tuple[float, float]] = []
    for epoch in range(1, epochs + 1):
        losses = [trainer.train_batch(real.to(trainer.device))
                  for real, _ in loader]
        avg_d, avg_g = (sum(v) / len(losses) for v in zip(*losses))
        history.append((avg_d, avg_g))
        print(f"{epoch:02d} D={avg_d:.3f} G={avg_g:.3f}")
        if epoch in GRID_EPOCHS or epoch == epochs:
            trainer.generate_grid(epoch)
    return history


def plot_losses(history: list[tuple[float, float]]) -> None:
    epochs = range(1, len(history) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [d for d, _ in history], label="Discriminator")
    plt.plot(epochs, [g for _, g in history], label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Mean batch loss")
    plt.legend()
    plt.savefig(OUT_DIR / "gan_losses.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'gan_losses.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = setup()
    loader = DataLoader(load_fashion_mnist(), batch_size=args.batch_size,
                        shuffle=True)
    trainer = GANTrainer(device)
    history = train(trainer, loader, epochs=args.epochs)
    plot_losses(history)
    torch.save(trainer.G.state_dict(), OUT_DIR / "fashion_generator.pt")
    print(f"Saved {OUT_DIR / 'fashion_generator.pt'}")

    print("\nEpoch | D loss | G loss")
    for epoch, (d, g) in enumerate(history, 1):
        if epoch in GRID_EPOCHS:
            print(f"{epoch:>5} | {d:.3f}  | {g:.3f}")


if __name__ == "__main__":
    main()
