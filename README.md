# FusionUNet — Whole Slide Image (WSI) Tissue Segmentation

A PyTorch inference pipeline for automated tissue segmentation on Whole Slide Images (WSIs) using **FusionUNet**—a dual-encoder segmentation architecture combining **DenseNet-121** and **MobileNetV2** backbones with gated feature fusion and concurrent spatial-and-channel Squeeze-and-Excitation (**scSE**) attention.

The pipeline reads multi-resolution WSI pyramid levels via OpenSlide at a target microns-per-pixel (MPP) resolution, performs overlapping tile inference with smooth feather blending and test-time augmentation (TTA), applies morphological post-processing, and exports binary masks, probability maps, visual overlays, and **QuPath-compatible GeoJSON** annotations in level-0 coordinates.

## Sample Result

<img width="2190" height="498" alt="image" src="https://github.com/user-attachments/assets/ee33bf13-cbd7-4fd3-8c21-4e5c4c41e758" />

---

## Features

- **Resolution-Agnostic Inference:** Automatically selects the optimal OpenSlide pyramid level and resamples tiles to match the model's training MPP (`--target-mpp`).
- **Seamless Tiling:** Uses linear ramp feather blending across overlapping tiles (`--overlap`) to eliminate grid artifacts.
- **Fast Execution:** Skips blank glass and out-of-bounds background tiles (`--skip-bg`), supports mixed-precision (`FP16`) inference on CUDA GPUs, and includes 4-pass flip Test-Time Augmentation (TTA).
- **Pathology-Ready Exports:**
  - `*_mask.png` — Binary tissue mask at target MPP.
  - `*_prob.png` — Raw tissue probability map (`0–255`).
  - `*_overlay.jpg` — Thumbnail with semi-transparent tissue mask and contour outline.
  - `*_tissue.geojson` — Polygon annotations with hole support in level-0 slide coordinates (ready to drag-and-drop into **QuPath**).
  - `*_summary.png` — 4-panel diagnostic figure (Thumbnail, Probability, Mask, Overlay).
  - `*_info.json` — Slide metadata, parameters used, and computed tissue area ($\text{mm}^2$).

---

## Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/SoroushOskouei/tissue-segmentation.git](https://github.com/SoroushOskouei/tissue-segmentation.git)
cd tissue-segmentation
```

### 2. Install Dependencies
```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm openslide-python openslide-bin huggingface_hub
```

### 3. Download Model Weights from Hugging Face
Pretrained model weights (`best_fusion_unet.pt`) are hosted on Hugging Face: **[soroush-oskouei/tissue_segmentation](https://huggingface.co/soroush-oskouei/tissue_segmentation)**

Download using the Hugging Face CLI:

```bash
hf download soroush-oskouei/tissue_segmentation best_fusion_unet.pt --local-dir weights
```

Or download via Python:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="soroush-oskouei/tissue_segmentation",
    filename="best_fusion_unet.pt",
    local_dir="weights",
)
```

---

## Usage

### Basic Inference
Run segmentation on a single whole slide image (`.svs`, `.ndpi`, `.tif`, `.mrxs`, etc.):

```bash
python infer.py \
  --wsi path/to/slide.svs \
  --checkpoint weights/best_fusion_unet.pt \
  --out-dir results/
```

### Quick Test with a Sample Slide
Use `--download-sample` to automatically fetch a public Aperio sample slide (`CMU-1.svs`) and run inference:

```bash
python infer.py \
  --download-sample \
  --checkpoint weights/best_fusion_unet.pt \
  --out-dir results/
```

### Advanced Options

```bash
python infer.py \
  --wsi path/to/slide.svs \
  --checkpoint weights/best_fusion_unet.pt \
  --out-dir results/ \
  --target-mpp 4.0 \
  --overlap 100 \
  --batch-size 16 \
  --num-workers 4 \
  --min-tissue-area 20000 \
  --min-hole-area 10000
```

### CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--wsi` | `None` | Path to the input Whole Slide Image. |
| `--download-sample` | `False` | Download and run on the sample `CMU-1.svs` slide if `--wsi` is not provided. |
| `--checkpoint` | `weights/best_fusion_unet.pt` | Path to the trained PyTorch model checkpoint (`.pt`). |
| `--out-dir` | `outputs` | Directory where masks, overlays, and GeoJSON files are saved. |
| `--target-mpp` | `4.0` | Target resolution in $\mu\text{m/px}$ matching training scale. |
| `--base-mpp-override` | `None` | Manually specify level-0 MPP if missing from slide metadata (e.g., `0.25`). |
| `--tile-size` | `None` | Tile size in pixels at target MPP (defaults to `img_size` stored in checkpoint). |
| `--overlap` | `100` | Overlap in pixels between adjacent tiles (must be `< tile_size / 2`). |
| `--batch-size` | `8` | Inference batch size. |
| `--num-workers` | `2` | Number of DataLoader worker processes. |
| `--threshold` | `None` | Probability cutoff in `[0, 1]` (defaults to tuned threshold in checkpoint). |
| `--no-tta` | `False` | Disable 4-pass flip Test-Time Augmentation for faster inference. |
| `--no-skip-bg` | `False` | Process all tiles instead of skipping blank glass/background tiles. |
| `--min-tissue-area` | `20000` | Remove disconnected tissue islands smaller than this area ($\mu\text{m}^2$). |
| `--min-hole-area` | `10000` | Fill interior holes smaller than this area ($\mu\text{m}^2$). |
| `--no-geojson` | `False` | Skip exporting QuPath-compatible `.geojson` polygons. |
| `--no-figure` | `False` | Skip saving the 4-panel `*_summary.png` visualization. |
| `--thumb-max-side` | `2048` | Maximum dimension in pixels for the output thumbnail and overlay. |
| `--device` | `auto` | Compute device (`auto`, `cuda`, `mps`, or `cpu`). |

---

## Acknowledgments & Data Attribution

The training data used to develop and validate the weights for this model were generated by **The Cancer Genome Atlas (TCGA)** Research Network ([https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)). We gratefully acknowledge the TCGA consortium, specimen donors, and contributing clinical sites for making these histopathology whole slide images publicly available.
