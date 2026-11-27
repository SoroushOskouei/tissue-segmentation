#!/usr/bin/env python3
"""
WSI Tissue Segmentation Inference Pipeline (FusionUNet).

Tiles a Whole Slide Image via OpenSlide at a target microns-per-pixel (MPP),
runs FusionUNet with optional Test-Time Augmentation (TTA), blends overlapping
tiles, applies morphological post-processing, and exports binary masks,
probability maps, overlays, summary figures, and QuPath-compatible GeoJSON.
"""

import argparse
import json
import math
import os
import time
import urllib.request

import cv2
import matplotlib

matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121, mobilenet_v2
from tqdm.auto import tqdm

cv2.setNumThreads(0)

SAMPLE_WSI_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"


# ----------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------
def conv_bn_relu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SCSE(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation block."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        h = max(c // r, 8)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, h, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, c, 1),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cse(x) + x * self.sse(x)


class GatedFusion(nn.Module):
    """Gated feature fusion between DenseNet and MobileNet feature maps."""

    def __init__(self, c_dense: int, c_mobile: int, c_out: int):
        super().__init__()
        self.pd = conv_bn_relu(c_dense, c_out, 1)
        self.pm = conv_bn_relu(c_mobile, c_out, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * c_out, c_out, 1), nn.Sigmoid())
        self.refine = nn.Sequential(conv_bn_relu(c_out, c_out, 3), SCSE(c_out))

    def forward(self, fd: torch.Tensor, fm: torch.Tensor) -> torch.Tensor:
        if fm.shape[-2:] != fd.shape[-2:]:
            fm = F.interpolate(fm, size=fd.shape[-2:], mode="bilinear", align_corners=False)
        d, m = self.pd(fd), self.pm(fm)
        g = self.gate(torch.cat([d, m], dim=1))
        return self.refine(g * d + (1.0 - g) * m)


class FusionEncoder(nn.Module):
    DENSE_CH = (64, 256, 512, 1024, 1024)
    MOBILE_CH = (16, 24, 32, 96, 320)

    def __init__(
        self,
        fused_ch=(64, 96, 128, 256, 384),
        pretrained: bool = False,
        memory_efficient: bool = False,
    ):
        super().__init__()
        dn = densenet121(weights=None, memory_efficient=memory_efficient).features
        mb = mobilenet_v2(weights=None).features
        self.d0 = nn.Sequential(dn.conv0, dn.norm0, dn.relu0)
        self.d1 = nn.Sequential(dn.pool0, dn.denseblock1)
        self.d2 = nn.Sequential(dn.transition1, dn.denseblock2)
        self.d3 = nn.Sequential(dn.transition2, dn.denseblock3)
        self.m0, self.m1, self.m2, self.m3, self.m4 = (
            mb[0:2],
            mb[2:4],
            mb[4:7],
            mb[7:14],
            mb[14:18],
        )
        self.fuse = nn.ModuleList(
            [GatedFusion(cd, cm, cf) for cd, cm, cf in zip(self.DENSE_CH, self.MOBILE_CH, fused_ch)]
        )

    def forward(self, x: torch.Tensor):
        d0 = self.d0(x)
        d1 = self.d1(d0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = F.avg_pool2d(d3, 2, ceil_mode=True)

        m0 = self.m0(x)
        m1 = self.m1(m0)
        m2 = self.m2(m1)
        m3 = self.m3(m2)
        m4 = self.m4(m3)

        return [f(d, m) for f, d, m in zip(self.fuse, (d0, d1, d2, d3, d4), (m0, m1, m2, m3, m4))]


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))
        self.att = SCSE(cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.att(self.conv(torch.cat([x, skip], dim=1)))


class FusionUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        fused_ch=(64, 96, 128, 256, 384),
        dec_ch=(256, 128, 64, 32, 16),
        pretrained: bool = False,
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.encoder = FusionEncoder(fused_ch, pretrained, memory_efficient)
        self.stem = nn.Sequential(conv_bn_relu(3, 16), conv_bn_relu(16, 16))
        self.center = nn.Sequential(conv_bn_relu(fused_ch[4], fused_ch[4]), SCSE(fused_ch[4]))
        self.dec4 = DecoderBlock(fused_ch[4], fused_ch[3], dec_ch[0])
        self.dec3 = DecoderBlock(dec_ch[0], fused_ch[2], dec_ch[1])
        self.dec2 = DecoderBlock(dec_ch[1], fused_ch[1], dec_ch[2])
        self.dec1 = DecoderBlock(dec_ch[2], fused_ch[0], dec_ch[3])
        self.dec0 = DecoderBlock(dec_ch[3], 16, dec_ch[4])
        self.head = nn.Conv2d(dec_ch[4], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3, f4 = self.encoder(x)
        s = self.stem(x)
        y = self.center(f4)
        y = self.dec4(y, f3)
        y = self.dec3(y, f2)
        y = self.dec2(y, f1)
        y = self.dec1(y, f0)
        y = self.dec0(y, s)
        return self.head(y)


# ----------------------------------------------------------------------
# WSI Helpers & Dataset
# ----------------------------------------------------------------------
def get_base_mpp(slide: openslide.OpenSlide, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    props = slide.properties
    for key in (openslide.PROPERTY_NAME_MPP_X, "aperio.MPP"):
        if key in props:
            return float(props[key])
    if "tiff.XResolution" in props and "tiff.ResolutionUnit" in props:
        res = float(props["tiff.XResolution"])
        unit = props["tiff.ResolutionUnit"].lower()
        if res > 0 and unit in ("centimeter", "cm"):
            return 1e4 / res
        if res > 0 and unit == "inch":
            return 25400.0 / res
    if openslide.PROPERTY_NAME_OBJECTIVE_POWER in props:
        mpp = 10.0 / float(props[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        print(f"WARNING: MPP estimated from objective power: {mpp:.3f} µm/px")
        return mpp
    raise ValueError("No MPP found in slide metadata. Pass --base-mpp-override (e.g., 0.25).")


def grid_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def blend_weight(tile: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((tile, tile), dtype=np.float32)
    ramp = np.ones(tile, dtype=np.float32)
    r = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
    ramp[:overlap] = r
    ramp[-overlap:] = r[::-1]
    return np.outer(ramp, ramp)


def is_background(rgb: np.ndarray) -> bool:
    """Detect blank glass or out-of-slide black padding conservatively."""
    return bool((rgb.min(axis=2) > 235).mean() > 0.995 or (rgb.max(axis=2) < 10).mean() > 0.995)


class WSITileDataset(Dataset):
    """Reads tiles from the best OpenSlide pyramid level and resamples to target MPP."""

    def __init__(self, path, coords, tile, scale, level, downsample, mean, std, skip_bg):
        self.path = path
        self.coords = coords
        self.tile = tile
        self.scale = scale
        self.level = level
        self.ds = downsample
        self.mean = mean
        self.std = std
        self.skip_bg = skip_bg
        self.slide = None  # Opened lazily per worker process

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.path)
        x, y = self.coords[idx]
        x0, y0 = int(round(x * self.scale)), int(round(y * self.scale))
        rs = max(1, int(math.ceil(self.tile * self.scale / self.ds)))
        rgba = np.array(self.slide.read_region((x0, y0), self.level, (rs, rs)))
        rgb = rgba[..., :3].copy()
        rgb[rgba[..., 3] == 0] = 255  # Transparent padding -> white background

        if rs != self.tile:
            interp = cv2.INTER_AREA if rs > self.tile else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (self.tile, self.tile), interpolation=interp)

        bg = bool(self.skip_bg and is_background(rgb))
        tensor = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(tensor.transpose(2, 0, 1)), x, y, bg


# ----------------------------------------------------------------------
# Prediction & Post-Processing
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_batch(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    tta: bool = True,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        p = torch.sigmoid(model(x).float())
        if tta:
            p += torch.sigmoid(model(x.flip(3)).float()).flip(3)
            p += torch.sigmoid(model(x.flip(2)).float()).flip(2)
            p += torch.sigmoid(model(x.flip(2, 3)).float()).flip(2, 3)
            p /= 4.0
    return p[:, 0]


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0:
        return mask.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def postprocess(
    prob: np.ndarray,
    thr: float,
    mpp: float,
    min_obj_um2: float,
    min_hole_um2: float,
) -> np.ndarray:
    m = (prob >= thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)
    px_area = mpp ** 2
    m = remove_small_components(m, int(min_obj_um2 / px_area))
    m = ~remove_small_components(~m, int(min_hole_um2 / px_area))
    return m


def mask_to_geojson(
    mask: np.ndarray,
    scale: float,
    simplify_px: float = 1.0,
    name: str = "Tissue",
) -> dict:
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    feats = []
    if hier is None:
        return {"type": "FeatureCollection", "features": feats}
    hier = hier[0]

    def ring(c):
        c = cv2.approxPolyDP(c, simplify_px, True)[:, 0, :].astype(np.float64) * scale
        if len(c) < 3:
            return None
        pts = c.round(1).tolist()
        pts.append(pts[0])
        return pts

    for i, (_, _, child, parent) in enumerate(hier):
        if parent != -1:
            continue
        outer = ring(contours[i])
        if outer is None:
            continue
        rings, j = [outer], child
        while j != -1:
            r = ring(contours[j])
            if r is not None:
                rings.append(r)
            j = hier[j][0]
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": rings},
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": name, "color": [0, 200, 0]},
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


# ----------------------------------------------------------------------
# CLI & Main Pipeline
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FusionUNet tissue segmentation on a Whole Slide Image (WSI)."
    )
    parser.add_argument("--wsi", type=str, default=None, help="Path to input WSI file.")
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Download and run inference on sample CMU-1.svs if --wsi is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_fusion_unet.pt",
        help="Path to trained FusionUNet checkpoint (.pt).",
    )
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory for output files.")
    parser.add_argument("--target-mpp", type=float, default=4.0, help="Target resolution in µm/px.")
    parser.add_argument(
        "--base-mpp-override",
        type=float,
        default=None,
        help="Override level-0 MPP if missing from slide metadata.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size in px at target MPP (defaults to checkpoint img_size).",
    )
    parser.add_argument("--overlap", type=int, default=100, help="Tile overlap in px (< tile/2).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold (defaults to checkpoint value or 0.5).",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable 4-pass flip TTA.")
    parser.add_argument("--no-skip-bg", action="store_true", help="Do not skip blank glass tiles.")
    parser.add_argument(
        "--min-tissue-area",
        type=float,
        default=20000.0,
        help="Minimum tissue component area in µm².",
    )
    parser.add_argument(
        "--min-hole-area",
        type=float,
        default=10000.0,
        help="Maximum interior hole area to fill in µm².",
    )
    parser.add_argument("--no-geojson", action="store_true", help="Disable GeoJSON export.")
    parser.add_argument("--no-figure", action="store_true", help="Disable 4-panel summary image.")
    parser.add_argument("--thumb-max-side", type=int, default=2048, help="Max thumbnail dimension.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    # Resolve input WSI
    wsi_path = args.wsi
    if wsi_path is None:
        if args.download_sample:
            os.makedirs("sample_data", exist_ok=True)
            wsi_path = os.path.join("sample_data", "CMU-1.svs")
            if not os.path.exists(wsi_path):
                print(f"Downloading sample WSI to {wsi_path} ...")
                urllib.request.urlretrieve(SAMPLE_WSI_URL, wsi_path)
        else:
            raise ValueError("Please provide --wsi <path_to_slide> or pass --download-sample.")

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    # Load model checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FusionUNet(**ckpt["model_kwargs"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tile_size = args.tile_size or ckpt["img_size"]
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    thr = args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5)
    score_key = "val_dice" if "val_dice" in ckpt else "val_f1"
    print(
        f"Loaded epoch {ckpt['epoch']} ({score_key}: {ckpt[score_key]:.4f}) | "
        f"device: {device} | tile: {tile_size}px | threshold: {thr:.3f}"
    )

    # Open WSI and compute grid coordinates
    t_start = time.time()
    slide = openslide.OpenSlide(wsi_path)
    W0, H0 = slide.dimensions
    base_mpp = get_base_mpp(slide, args.base_mpp_override)
    target_mpp = args.target_mpp
    scale = target_mpp / base_mpp
    if scale < 1.0:
        print("WARNING: target MPP is finer than base scan MPP; tiles will be upsampled.")

    level = slide.get_best_level_for_downsample(max(scale, 1.0))
    ds = slide.level_downsamples[level]
    Wt, Ht = int(math.ceil(W0 / scale)), int(math.ceil(H0 / scale))

    ov = args.overlap
    if not (0 <= ov < tile_size // 2):
        raise ValueError(f"Overlap ({ov}) must be >= 0 and < tile_size // 2 ({tile_size // 2}).")
    stride = tile_size - ov
    coords = [
        (x, y)
        for y in grid_positions(Ht, tile_size, stride)
        for x in grid_positions(Wt, tile_size, stride)
    ]

    print(
        f"Slide {W0}x{H0} @ {base_mpp:.4f} µm/px | levels: {slide.level_count} | "
        f"reading level {level} (ds {ds:.2f})"
    )
    print(f"Target {target_mpp} µm/px -> canvas {Wt}x{Ht} | {len(coords)} tiles, overlap {ov}px")

    dataset = WSITileDataset(
        wsi_path,
        coords,
        tile_size,
        scale,
        level,
        ds,
        mean,
        std,
        skip_bg=not args.no_skip_bg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    weight = blend_weight(tile_size, ov)
    prob_sum = np.zeros((Ht, Wt), dtype=np.float32)
    w_sum = np.zeros((Ht, Wt), dtype=np.float32)
    n_skipped = 0

    for imgs, xs, ys, bg in tqdm(loader, desc="Running inference"):
        bg_np = bg.numpy().astype(bool)
        probs = np.zeros((imgs.size(0), tile_size, tile_size), dtype=np.float32)
        keep = ~bg_np
        n_skipped += int(bg_np.sum())
        if keep.any():
            batch_x = imgs[torch.from_numpy(keep)].to(device, non_blocking=True)
            probs[keep] = (
                predict_batch(model, batch_x, device=device, use_amp=use_amp, tta=not args.no_tta)
                .cpu()
                .numpy()
            )
        for p, x, y in zip(probs, xs.tolist(), ys.tolist()):
            h, w = min(tile_size, Ht - y), min(tile_size, Wt - x)
            prob_sum[y : y + h, x : x + w] += p[:h, :w] * weight[:h, :w]
            w_sum[y : y + h, x : x + w] += weight[:h, :w]

    prob = prob_sum / np.maximum(w_sum, 1e-6)
    del prob_sum, w_sum

    mask = postprocess(prob, thr, target_mpp, args.min_tissue_area, args.min_hole_area)
    tissue_mm2 = float(mask.sum() * (target_mpp ** 2) / 1e6)
    elapsed = time.time() - t_start
    print(
        f"Done in {elapsed:.1f}s | skipped {n_skipped}/{len(coords)} background tiles | "
        f"tissue area: {tissue_mm2:.2f} mm² ({100.0 * mask.mean():.1f}% of slide)"
    )

    # Save outputs
    stem = os.path.splitext(os.path.basename(wsi_path))[0]
    tag = f"{stem}_mpp{target_mpp:g}"

    cv2.imwrite(os.path.join(args.out_dir, f"{tag}_mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.out_dir, f"{tag}_prob.png"), (prob * 255).astype(np.uint8))

    thumb = np.array(
        slide.get_thumbnail((args.thumb_max_side, args.thumb_max_side)).convert("RGB")
    )
    th, tw = thumb.shape[:2]
    mask_t = cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
    prob_t = cv2.resize(prob, (tw, th), interpolation=cv2.INTER_AREA)

    overlay = thumb.copy()
    overlay[mask_t > 0] = (0.6 * overlay[mask_t > 0] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 120, 0), 2)
    cv2.imwrite(
        os.path.join(args.out_dir, f"{tag}_overlay.jpg"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    if not args.no_geojson:
        gj = mask_to_geojson(mask, scale)
        gj_path = os.path.join(args.out_dir, f"{tag}_tissue.geojson")
        with open(gj_path, "w", encoding="utf-8") as f:
            json.dump(gj, f)
        print(f"GeoJSON: {len(gj['features'])} tissue polygons saved (level-0 coordinates)")

    if not args.no_figure:
        fig, ax = plt.subplots(1, 4, figsize=(22, 6))
        ax[0].imshow(thumb)
        ax[0].set_title("Thumbnail")
        ax[1].imshow(prob_t, cmap="magma", vmin=0, vmax=1)
        ax[1].set_title("Tissue Probability")
        ax[2].imshow(mask_t, cmap="gray")
        ax[2].set_title(f"Mask (thr {thr:.2f})")
        ax[3].imshow(overlay)
        ax[3].set_title("Overlay")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{tag}_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    info = {
        "slide": os.path.abspath(wsi_path),
        "base_mpp": base_mpp,
        "target_mpp": target_mpp,
        "level_used": level,
        "canvas_wh": [Wt, Ht],
        "tile_size": tile_size,
        "overlap": ov,
        "threshold": float(thr),
        "tissue_area_mm2": tissue_mm2,
        "runtime_sec": round(elapsed, 2),
    }
    with open(os.path.join(args.out_dir, f"{tag}_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    slide.close()
    print(f"All outputs saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
