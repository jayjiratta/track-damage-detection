"""
PaDiM (Patch Distribution Modeling) anomaly detection
======================================================
Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
Detection and Localization" (ICPR 2020).

  1. Multi-level feature maps from a frozen CNN are brought to the finest
     level's resolution (nearest upsampling) and concatenated channel-wise ->
     one embedding per spatial patch position.
  2. A fixed random subset of EMBED_DIM channels is kept (the paper's
     dimensionality reduction; same fixed subset at fit and score time).
  3. Fit: for every patch position, a multivariate Gaussian (mean, covariance
     + eps*I) over that position's embeddings across all training images.
  4. Score: Mahalanobis distance of a test image's embedding to its
     position's Gaussian -> anomaly map, upsampled to the input image size and
     Gaussian-smoothed.

Two feature sources (backbones):

  resnet18   method 1 (separate backbones): an ImageNet-pretrained ResNet-18,
             layer1/2/3 (stride 4/8/16, 64+128+256 ch) of the image resized to
             448x256, run next to the YOLOv26n track model.
  shared_r18 method 2 (shared backbone): the track model is a YOLO26-seg
             neck/head on a FROZEN ImageNet ResNet-18
             (train/yolo26-seg-resnet18.yaml); PaDiM takes layer1/2/3 of that
             same backbone, captured with forward hooks during the forward pass
             that produces the track mask, so one CNN pass serves both. YOLO
             letterboxes 1280x720 to 640x384; the padding rows are cropped off
             the anomaly map before resizing to the image size.

No labels are used anywhere -- training images are assumed to be (mostly)
normal/undamaged track surface.

Assumption specific to this use: PaDiM models each patch *position*
separately, so it relies on images being roughly aligned. That holds here
because every frame comes from the same fixed camera mount driving along the
track, but it would not hold for arbitrarily framed photos. It also means a
normal pattern that is rare at a given position (e.g. a painted marking that
appears there in only a few training frames) scores as anomalous.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

INPUT_W, INPUT_H = 448, 256       # resnet18 input, ~same aspect as the 1280x720 frames
EMBED_DIM = 100                   # channels kept (paper: d=100 for ResNet-18)
COV_EPS = 0.01                    # covariance regularizer (paper value)
RANDOM_SEED = 1024
SMOOTH_SIGMA = 4

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# shared_r18: children of the TorchVision backbone module (model.model[0].m):
# conv1, bn1, relu, maxpool, layer1 (4), layer2 (5), layer3 (6), layer4 (7)
SHARED_R18_LAYERS = (4, 5, 6)
SHARED_R18_STRIDE = {4: 4, 5: 8, 6: 16}
SHARED_R18_GRID_STRIDE = 8   # 48x80 patch grid: same quality as 96x160 at a quarter of the model size


class YoloFeatureHook:
    """Keeps the outputs of chosen children of the track model's TorchVision
    ResNet backbone (model.model[0].m) from the latest forward pass."""

    def __init__(self, yolo_model, layers=SHARED_R18_LAYERS):
        self.layers = tuple(layers)
        self.feats: dict[int, torch.Tensor] = {}
        # once predict() has run, ultralytics runs its own (fused) copy of the
        # network, so hook that one; before that, the model it will wrap
        pred = getattr(yolo_model, "predictor", None)
        net = pred.model.model if pred is not None and getattr(pred, "model", None) is not None else yolo_model.model
        mods = net.model[0].m
        for i in self.layers:
            mods[i].register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(_module, _inp, out):
            self.feats[i] = out.detach()
        return fn

    def get(self) -> list[torch.Tensor]:
        return [self.feats[i] for i in self.layers]


class PaDiM:
    def __init__(self, backbone: str = "resnet18", device: str | None = None, yolo=None,
                 grid_stride: int | None = None):
        """backbone="shared_r18" needs `yolo`: the ultralytics YOLO track model
        whose forward pass is shared (hooks are registered on it).
        grid_stride: its patch grid (default SHARED_R18_GRID_STRIDE); a coarser
        grid than a layer's own average-pools that layer."""
        if backbone not in ("resnet18", "shared_r18"):
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone_name = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if backbone == "resnet18":
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = net.eval().to(self.device)
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            n_ch = 64 + 128 + 256
        else:
            if yolo is None:
                raise ValueError("backbone='shared_r18' needs the frozen-ResNet-18 YOLO track model")
            self.yolo = yolo
            self.yolo_layers = SHARED_R18_LAYERS
            self.hook = YoloFeatureHook(yolo, self.yolo_layers)
            self.grid_stride = SHARED_R18_GRID_STRIDE
            n_ch = 64 + 128 + 256
        if backbone != "resnet18" and grid_stride:
            self.grid_stride = grid_stride

        gen = torch.Generator().manual_seed(RANDOM_SEED)
        self.channel_idx = torch.randperm(n_ch, generator=gen)[:EMBED_DIM].to(self.device)

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

    def _combine(self, feats: list[torch.Tensor]) -> torch.Tensor:
        size = feats[0].shape[-2:]               # finest level's resolution (paper)
        if self.backbone_name != "resnet18":
            s0 = SHARED_R18_STRIDE[self.yolo_layers[0]]
            size = (size[0] * s0 // self.grid_stride, size[1] * s0 // self.grid_stride)
        out = []
        for f in feats:
            f = f.float()
            if f.shape[-2] > size[0]:
                f = F.adaptive_avg_pool2d(f, size)
            elif f.shape[-2] < size[0]:
                f = F.interpolate(f, size=size, mode="nearest")
            out.append(f)
        return torch.cat(out, dim=1).index_select(1, self.channel_idx)

    @torch.no_grad()
    def _embed(self, imgs_bgr: list[np.ndarray]) -> torch.Tensor:
        """Returns (B, EMBED_DIM, h, w) patch embeddings."""
        if self.backbone_name == "shared_r18":
            self.yolo.predict(imgs_bgr, verbose=False, retina_masks=True)   # hooks capture the features
            return self._combine(self.hook.get())
        b = self.backbone
        x = self._preprocess(imgs_bgr)
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        f1 = b.layer1(x)
        f2 = b.layer2(f1)
        f3 = b.layer3(f2)
        return self._combine([f1, f2, f3])

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
    def score(self, img_bgr: np.ndarray, feats: list[torch.Tensor] | None = None) -> np.ndarray:
        """Anomaly map (float32, same HxW as img_bgr): Mahalanobis distance per pixel.
        For backbone="shared_r18", pass `feats` (YoloFeatureHook.get() after the ROI
        model's forward on this frame) to reuse that forward pass."""
        if self.mean is None:
            raise RuntimeError("PaDiM model is not fitted/loaded")
        if feats is not None:
            emb = self._combine(feats)[0]
        else:
            emb = self._embed([img_bgr])[0]                      # (d, h, w)
        d, h, w = emb.shape
        diff = emb.reshape(d, h * w).T - self.mean               # (P, d)
        m = torch.einsum("pi,pij,pj->p", diff, self.inv_cov, diff).clamp_min(0).sqrt()
        amap = m.reshape(h, w).cpu().numpy()
        H, W = img_bgr.shape[:2]
        if self.backbone_name == "shared_r18":
            amap = self._crop_letterbox(amap, H, W)
        amap = cv2.resize(amap, (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(amap, (0, 0), SMOOTH_SIGMA)

    def _crop_letterbox(self, amap: np.ndarray, H: int, W: int) -> np.ndarray:
        """Drop the rows/cols that cover YOLO's letterbox padding (centred padding)."""
        h, w = amap.shape
        stride = self.grid_stride
        in_h, in_w = h * stride, w * stride
        r = min(in_h / H, in_w / W)
        pad_y, pad_x = (in_h - H * r) / 2 / stride, (in_w - W * r) / 2 / stride
        big = cv2.resize(amap, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        y0, x0 = int(round(pad_y * stride)), int(round(pad_x * stride))
        return big[y0:in_h - y0, x0:in_w - x0]

    # ------------------------------------------------------- spatial resolution
    def _input_scale(self, H: int, W: int) -> tuple[float, float]:
        """Network-input pixels per image pixel (x, y)."""
        if self.backbone_name == "resnet18":
            return INPUT_W / W, INPUT_H / H
        gh, gw = self.grid_hw
        r = min(gh * self.grid_stride / H, gw * self.grid_stride / W)     # YOLO letterbox scale
        return r, r

    def cell_px(self, H: int, W: int) -> tuple[float, float]:
        """(w, h) in image pixels of one patch of the anomaly map: the smallest
        area PaDiM gives its own score to."""
        sx, sy = self._input_scale(H, W)
        stride = self.grid_stride if self.backbone_name != "resnet18" else SHARED_R18_STRIDE[4]
        return stride / sx, stride / sy

    def coarse_cell_px(self, H: int, W: int) -> tuple[float, float]:
        """(w, h) in image pixels of one cell of the coarsest feature map used
        (layer3, stride 16): structures closer than this share one layer3 feature."""
        sx, sy = self._input_scale(H, W)
        return SHARED_R18_STRIDE[6] / sx, SHARED_R18_STRIDE[6] / sy

    # --------------------------------------------------------------- save/load
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"mean": self.mean.cpu(), "inv_cov": self.inv_cov.cpu(),
                 "grid_hw": self.grid_hw, "n_train": self.n_train,
                 "channel_idx": self.channel_idx.cpu(), "embed_dim": len(self.channel_idx),
                 "backbone": self.backbone_name}
        if self.backbone_name == "resnet18":
            state["input_wh"] = (INPUT_W, INPUT_H)
        else:
            state["yolo_layers"] = self.yolo_layers
            state["grid_stride"] = self.grid_stride
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None, yolo=None) -> "PaDiM":
        state = torch.load(path, map_location="cpu")
        backbone = state.get("backbone", "resnet18")
        model = cls(backbone, device, yolo=yolo, grid_stride=state.get("grid_stride"))
        if state["embed_dim"] != len(model.channel_idx) or (backbone == "resnet18" and tuple(state["input_wh"]) != (INPUT_W, INPUT_H)):
            raise ValueError("saved PaDiM was fitted with different INPUT/EMBED settings")
        model.mean = state["mean"].to(model.device)
        model.inv_cov = state["inv_cov"].to(model.device)
        model.channel_idx = state["channel_idx"].to(model.device)
        model.grid_hw = tuple(state["grid_hw"])
        model.n_train = state["n_train"]
        return model
