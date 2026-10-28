from pathlib import Path
import sys
import uuid
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
PACK_ROOT = BASE_DIR / "models" / "final_hybrid_fabric_artifacts"
INFER_DIR = PACK_ROOT / "inference"

if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from final_hybrid_infer import HybridInfer  # noqa: E402


_INFER_CACHE = {}

# Type6 özel branch: RealNet/DeSTSeg istemez, sadece görüntü üzerinden bright/contrast mask üretir.
TYPE6_BRIGHT_THRESHOLD = 0.85
TYPE6_BRIGHT_MIN_AREA = 3
TYPE6_IMAGE_SIZE = 256


def is_final_hybrid_ready() -> bool:
    required = [
        PACK_ROOT / "configs" / "final_hybrid_config.json",
        PACK_ROOT / "inference" / "final_hybrid_infer.py",
        PACK_ROOT / "checkpoints" / "type9cam2_token_autoencoder.pt",
        PACK_ROOT / "checkpoints" / "vit_small_patch14_dinov2_state_dict.pt",
        PACK_ROOT / "banks" / "type1cam1_dino_bank.pt",
        PACK_ROOT / "banks" / "type2cam2_dino_bank.pt",
        PACK_ROOT / "banks" / "type3cam1_dino_bank.pt",
        PACK_ROOT / "banks" / "type4cam2_dino_bank.pt",
        PACK_ROOT / "banks" / "type5cam2_dino_bank.pt",
        PACK_ROOT / "banks" / "type7cam2_dino_bank.pt",
        PACK_ROOT / "banks" / "type8cam1_dino_bank.pt",
        PACK_ROOT / "banks" / "type10cam2_dino_bank.pt",
    ]
    return all(p.exists() for p in required)


def get_final_hybrid_infer(device: Optional[str] = None) -> HybridInfer:
    key = device or "auto"
    if key not in _INFER_CACHE:
        _INFER_CACHE[key] = HybridInfer(pack_root=str(PACK_ROOT), device=device)
    return _INFER_CACHE[key]


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(x.min())) / max(1e-8, float(x.max() - x.min()))


def _component_filter(mask: np.ndarray, min_area: int = 3) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 1
    return out


def _colorize_heatmap(score: np.ndarray) -> Image.Image:
    score = np.clip(score, 0, 1).astype(np.float32)
    color = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))


def _overlay_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    img = img.convert("RGB").resize((TYPE6_IMAGE_SIZE, TYPE6_IMAGE_SIZE))
    img_np = np.array(img).astype(np.float32)
    red = np.zeros_like(img_np)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.55
    out = img_np * (1 - alpha) + red * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _border_fraction(mask_bin: np.ndarray) -> float:
    mask_bin = (mask_bin > 0).astype(np.uint8)
    total = int(mask_bin.sum())
    if total == 0:
        return 0.0

    h, w = mask_bin.shape[:2]
    border_px = max(8, int(min(h, w) * 0.08))
    border = np.zeros_like(mask_bin, dtype=np.uint8)
    border[:border_px, :] = 1
    border[-border_px:, :] = 1
    border[:, :border_px] = 1
    border[:, -border_px:] = 1
    return float((mask_bin * border).sum() / max(1, total))


def _mask_stats(mask_img: Image.Image) -> Dict:
    mask_np = np.array(mask_img.convert("L"))
    mask_bin = (mask_np > 0).astype(np.uint8)

    h, w = mask_bin.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    boxes = []
    largest_bbox = None
    largest_area = 0

    for lab in range(1, num):
        x, y, bw, bh, area = stats[lab]
        area = int(area)
        if area <= 0:
            continue

        boxes.append((int(x), int(y), int(bw), int(bh), area))
        if area > largest_area:
            largest_area = area
            largest_bbox = (int(x), int(y), int(bw), int(bh))

    pred_pixels = int(mask_bin.sum())
    mask_area_fraction = float(pred_pixels / max(1, h * w))
    border_fraction = _border_fraction(mask_bin)

    if largest_bbox is None:
        box_mode = "none"
    elif len(boxes) >= 6 or mask_area_fraction >= 0.12:
        box_mode = "union_only"
    else:
        box_mode = "components"

    return {
        "mask_np": mask_bin,
        "region_count": len(boxes),
        "largest_bbox": largest_bbox,
        "pred_pixels": pred_pixels,
        "mask_area_fraction": mask_area_fraction,
        "border_fraction": border_fraction,
        "box_mode": box_mode,
    }


def _scale_bbox_to_image(bbox, from_size, to_size):
    if bbox is None:
        return None
    from_w, from_h = from_size
    to_w, to_h = to_size
    x, y, w, h = bbox
    sx = to_w / max(1, from_w)
    sy = to_h / max(1, from_h)
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )


def _draw_bbox_on_overlay(overlay_img: Image.Image, bbox):
    if bbox is None:
        return overlay_img

    rgb = np.array(overlay_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = bbox
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _crop_from_bbox(
    pil_image: Image.Image,
    bbox,
    bbox_canvas_size=(TYPE6_IMAGE_SIZE, TYPE6_IMAGE_SIZE),
    pad: int = 8,
):
    if bbox is None:
        return None

    img = pil_image.convert("RGB")
    w_img, h_img = img.size

    scaled_bbox = _scale_bbox_to_image(
        bbox=bbox,
        from_size=bbox_canvas_size,
        to_size=(w_img, h_img),
    )
    if scaled_bbox is None:
        return None

    x, y, w, h = scaled_bbox
    sx = w_img / max(1, bbox_canvas_size[0])
    sy = h_img / max(1, bbox_canvas_size[1])
    pad_x = int(round(pad * sx))
    pad_y = int(round(pad * sy))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2))


def _run_type6_bright_contrast_heatmap(
    pil_image: Image.Image,
    out_root: Optional[str] = None,
) -> Dict:
    if out_root is None:
        out_root = BASE_DIR / "runtime_outputs" / "final_hybrid"
    else:
        out_root = Path(out_root)

    upload_dir = BASE_DIR / "runtime_uploads" / "final_hybrid"
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    image_path = upload_dir / f"type6cam2_{uid}.png"
    pil_image.convert("RGB").save(image_path)

    out_dir = out_root / f"type6cam2_bright_contrast_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_256 = pil_image.convert("RGB").resize((TYPE6_IMAGE_SIZE, TYPE6_IMAGE_SIZE))
    arr = np.array(img_256).astype(np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
    pos = np.maximum(gray - bg, 0)
    absdiff = np.abs(gray - bg)
    white = np.maximum(gray - 0.45, 0)

    score = 0.65 * _norm01(pos) + 0.25 * _norm01(absdiff) + 0.10 * _norm01(white)
    score = _norm01(score)

    raw_mask = (score >= TYPE6_BRIGHT_THRESHOLD).astype(np.uint8)
    mask = _component_filter(raw_mask, min_area=TYPE6_BRIGHT_MIN_AREA)

    heatmap_img = _colorize_heatmap(score)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    pred_overlay_img = _overlay_mask(img_256, mask)

    stats = _mask_stats(mask_img)
    marked_img = _draw_bbox_on_overlay(pred_overlay_img, stats["largest_bbox"])
    defect_crop = _crop_from_bbox(
        pil_image,
        stats["largest_bbox"],
        bbox_canvas_size=(TYPE6_IMAGE_SIZE, TYPE6_IMAGE_SIZE),
    )

    heatmap_path = out_dir / "heatmap_color.png"
    mask_path = out_dir / "pred_mask.png"
    overlay_path = out_dir / "pred_overlay.png"
    score_path = out_dir / "heatmap_score.npy"

    heatmap_img.save(heatmap_path)
    mask_img.save(mask_path)
    pred_overlay_img.save(overlay_path)
    np.save(score_path, score.astype(np.float32))

    guard_warnings = []
    ood_status = "ok"

    if stats["pred_pixels"] == 0:
        guard_warnings.append("score_mask_inconsistency")
        ood_status = "warn"
    if stats["mask_area_fraction"] > 0.35:
        guard_warnings.append("excessive_mask_area")
        ood_status = "warn"
    if stats["border_fraction"] > 0.50:
        guard_warnings.append("border_dominant_activation")
        ood_status = "warn"

    raw_result = {
        "branch": "type6_bright_contrast",
        "anomaly_score": float(score.max()),
        "pred_pixels": int(stats["pred_pixels"]),
        "outputs": {
            "heatmap_color": str(heatmap_path),
            "pred_mask": str(mask_path),
            "pred_overlay": str(overlay_path),
            "score_npy": str(score_path),
        },
        "extra": {
            "config": {
                "threshold": TYPE6_BRIGHT_THRESHOLD,
                "min_area": TYPE6_BRIGHT_MIN_AREA,
                "method": "bright_contrast_local",
            }
        },
    }

    return {
        "heatmap_overlay": heatmap_img,
        "raw_heatmap_overlay": heatmap_img,
        "marked_image": marked_img,
        "binary_mask": mask_img,
        "region_count": stats["region_count"],
        "largest_bbox": stats["largest_bbox"],
        "defect_crop": defect_crop,
        "heatmap_mode": "type6_bright_contrast",
        "display_heatmap_mode": "type6_bright_contrast_heatmap",
        "pixel_threshold": float(TYPE6_BRIGHT_THRESHOLD),
        "seed_threshold": 0.0,
        "mask_area_fraction": stats["mask_area_fraction"],
        "border_fraction": stats["border_fraction"],
        "box_mode": stats["box_mode"],
        "ood_status": ood_status,
        "guard_warnings": guard_warnings,
        "final_hybrid_result": raw_result,
        "final_hybrid_out_dir": str(out_dir),
        "final_hybrid_branch": "type6_bright_contrast",
        "final_hybrid_anomaly_score": float(score.max()),
        "final_hybrid_pred_pixels": int(stats["pred_pixels"]),
    }


def run_final_hybrid_heatmap(
    pil_image: Image.Image,
    type_name: str,
    out_root: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    PatchCore good/defect kararına dokunmaz; sadece heatmap/mask görsellerini üretir.

    Final karar:
    - type6cam2: özel bright/contrast branch
    - diğer typelar: eski Final Hybrid sistemi
    """
    type_name = str(type_name)

    if type_name == "type6cam2":
        return _run_type6_bright_contrast_heatmap(
            pil_image=pil_image,
            out_root=out_root,
        )

    if not is_final_hybrid_ready():
        raise FileNotFoundError(f"Final hybrid pack eksik görünüyor: {PACK_ROOT}")

    if out_root is None:
        out_root = BASE_DIR / "runtime_outputs" / "final_hybrid"
    else:
        out_root = Path(out_root)

    upload_dir = BASE_DIR / "runtime_uploads" / "final_hybrid"
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    image_path = upload_dir / f"{type_name}_{uid}.png"
    pil_image.convert("RGB").save(image_path)

    out_dir = out_root / f"{type_name}_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    infer = get_final_hybrid_infer(device=device)
    raw_result = infer.run(
        image_path=str(image_path),
        type_name=type_name,
        out_dir=str(out_dir),
    )

    outputs = raw_result["outputs"]
    heatmap_img = Image.open(outputs["heatmap_color"]).convert("RGB")
    pred_overlay_img = Image.open(outputs["pred_overlay"]).convert("RGB")
    mask_img = Image.open(outputs["pred_mask"]).convert("L")

    stats = _mask_stats(mask_img)
    marked_img = _draw_bbox_on_overlay(pred_overlay_img, stats["largest_bbox"])
    defect_crop = _crop_from_bbox(
        pil_image,
        stats["largest_bbox"],
        bbox_canvas_size=mask_img.size,
    )

    cfg = raw_result.get("extra", {}).get("config", {})
    pixel_threshold = (
        cfg.get("threshold")
        if cfg.get("threshold") is not None
        else cfg.get("dino_threshold")
        if cfg.get("dino_threshold") is not None
        else cfg.get("class_threshold")
        if cfg.get("class_threshold") is not None
        else cfg.get("weighted_threshold")
        if cfg.get("weighted_threshold") is not None
        else 0.0
    )

    guard_warnings = []
    ood_status = "ok"

    if stats["pred_pixels"] == 0:
        guard_warnings.append("score_mask_inconsistency")
        ood_status = "warn"
    if stats["mask_area_fraction"] > 0.35:
        guard_warnings.append("excessive_mask_area")
        ood_status = "warn"
    if stats["border_fraction"] > 0.50:
        guard_warnings.append("border_dominant_activation")
        ood_status = "warn"

    return {
        "heatmap_overlay": heatmap_img,
        "raw_heatmap_overlay": heatmap_img,
        "marked_image": marked_img,
        "binary_mask": mask_img,
        "region_count": stats["region_count"],
        "largest_bbox": stats["largest_bbox"],
        "defect_crop": defect_crop,
        "heatmap_mode": f"final_hybrid_{raw_result.get('branch', '-')}",
        "display_heatmap_mode": "final_hybrid_heatmap_color",
        "pixel_threshold": float(pixel_threshold or 0.0),
        "seed_threshold": 0.0,
        "mask_area_fraction": stats["mask_area_fraction"],
        "border_fraction": stats["border_fraction"],
        "box_mode": stats["box_mode"],
        "ood_status": ood_status,
        "guard_warnings": guard_warnings,
        "final_hybrid_result": raw_result,
        "final_hybrid_out_dir": str(out_dir),
        "final_hybrid_branch": raw_result.get("branch"),
        "final_hybrid_anomaly_score": raw_result.get("anomaly_score"),
        "final_hybrid_pred_pixels": raw_result.get("pred_pixels"),
    }
