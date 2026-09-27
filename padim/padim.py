"""
PaDiM (Patch Distribution Modeling) anomaly detection
======================================================
Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
Detection and Localization" (ICPR 2020).

  1. A frozen ImageNet-pretrained ResNet-18 extracts layer1/2/3 feature maps.
     Layer2/3 are upsampled (nearest) to layer1's resolution and concatenated
     channel-wise -> one 448-d embedding per spatial patch position.
  2. A fixed random subset of EMBED_DIM channels is kept (the paper's
     dimensionality reduction; same fixed subset at fit and score time).
  3. Fit: for every patch position, a multivariate Gaussian (mean, covariance
     + eps*I) over that position's embeddings across all training images.
  4. Score: Mahalanobis distance of a test image's embedding to its
     position's Gaussian -> anomaly map, upsampled to the input image size and
     Gaussian-smoothed.

No labels are used anywhere -- training images are assumed to be (mostly)
normal/undamaged track surface.

Assumption specific to this use: PaDiM models each patch *position*
separately, so it relies on images being roughly aligned. That holds here
because every frame comes from the same fixed camera mount walking the same
track (the track occupies the same image region frame to frame), but it
would not hold for arbitrarily framed photos.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

INPUT_W, INPUT_H = 448, 256       # ~same aspect as the 1280x720 source frames
EMBED_DIM = 100                   # of 448 concatenated channels (paper: d=100 for ResNet-18)
COV_EPS = 0.01                    # covariance regularizer (paper value)
RANDOM_SEED = 1024
SMOOTH_SIGMA = 4

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PaDiM:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        gen = torch.Generator().manual_seed(RANDOM_SEED)
        self.channel_idx = torch.randperm(64 + 128 + 256, generator=gen)[:EMBED_DIM].to(self.device)

        self.mean: torch.Tensor | None = None       # (P, d)
        self.inv_cov: torch.Tensor | None = None    # (P, d, d)
        self.grid_hw: tuple[int, int] | None = None
        self.n_train = 0

    # ---------------------------------------------------------------- features
    def _preprocess(self, imgs_bgr: list[np.ndarray]) -> torch.Tensor:
        batch = []
        for img in imgs_bgr:
            rgb = cv2.cvtColor(cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append((rgb - IMAGENET_MEAN) / IMAGENET_STD)
        return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(self.device)

    @torch.no_grad()
    def _embed(self, imgs_bgr: list[np.ndarray]) -> torch.Tensor:
        """Returns (B, EMBED_DIM, h, w) patch embeddings."""
        b = self.backbone
        x = self._preprocess(imgs_bgr)
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        f1 = b.layer1(x)
        f2 = b.layer2(f1)
        f3 = b.layer3(f2)
        size = f1.shape[-2:]
        emb = torch.cat([f1,
                         F.interpolate(f2, size=size, mode="nearest"),
                         F.interpolate(f3, size=size, mode="nearest")], dim=1)
        return emb.index_select(1, self.channel_idx)

    # --------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, image_paths: list[Path], batch_size: int = 16):
        s1 = s2 = None
        n = 0
        for i in range(0, len(image_paths), batch_size):
            imgs = [cv2.imread(str(p)) for p in image_paths[i:i + batch_size]]
            emb = self._embed(imgs)                              # (B, d, h, w)
            B, d, h, w = emb.shape
            e = emb.reshape(B, d, h * w).permute(2, 0, 1)        # (P, B, d)
            if s1 is None:
                s1 = torch.zeros(h * w, d, device=self.device, dtype=torch.float64)
                s2 = torch.zeros(h * w, d, d, device=self.device, dtype=torch.float64)
                self.grid_hw = (h, w)
            e64 = e.double()
            s1 += e64.sum(dim=1)
            s2 += torch.einsum("pbi,pbj->pij", e64, e64)
            n += B
            print(f"  embedded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

        mean = s1 / n
        cov = (s2 - n * torch.einsum("pi,pj->pij", mean, mean)) / (n - 1)
        cov += COV_EPS * torch.eye(cov.shape[-1], device=self.device, dtype=torch.float64)
        self.mean = mean.float()
        self.inv_cov = torch.linalg.inv(cov).float()
        self.n_train = n

    # ------------------------------------------------------------------- score
    @torch.no_grad()
    def score(self, img_bgr: np.ndarray) -> np.ndarray:
        """Anomaly map (float32, same HxW as img_bgr): Mahalanobis distance per pixel."""
        if self.mean is None:
            raise RuntimeError("PaDiM model is not fitted/loaded")
        emb = self._embed([img_bgr])[0]                          # (d, h, w)
        d, h, w = emb.shape
        diff = emb.reshape(d, h * w).T - self.mean               # (P, d)
        m = torch.einsum("pi,pij,pj->p", diff, self.inv_cov, diff).clamp_min(0).sqrt()
        amap = m.reshape(h, w).cpu().numpy()
        H, W = img_bgr.shape[:2]
        amap = cv2.resize(amap, (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(amap, (0, 0), SMOOTH_SIGMA)

    # --------------------------------------------------------------- save/load
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean.cpu(), "inv_cov": self.inv_cov.cpu(),
                    "grid_hw": self.grid_hw, "n_train": self.n_train,
                    "channel_idx": self.channel_idx.cpu(),
                    "input_wh": (INPUT_W, INPUT_H), "embed_dim": EMBED_DIM}, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "PaDiM":
        model = cls(device)
        state = torch.load(path, map_location=model.device)
        if tuple(state["input_wh"]) != (INPUT_W, INPUT_H) or state["embed_dim"] != EMBED_DIM:
            raise ValueError("saved PaDiM was fitted with different INPUT/EMBED settings")
        model.mean = state["mean"].to(model.device)
        model.inv_cov = state["inv_cov"].to(model.device)
        model.channel_idx = state["channel_idx"].to(model.device)
        model.grid_hw = tuple(state["grid_hw"])
        model.n_train = state["n_train"]
        return model
