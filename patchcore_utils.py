import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from anomalib.models import Patchcore


IMAGE_SIZE = (256, 256)


def load_thresholds_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ckpt_path(models_root: str, type_name: str) -> str:
    ckpt_path = (
        Path(models_root)
        / f"{type_name}_wr50_256_l23"
        / "Patchcore"
        / f"itd_{type_name}"
        / "v0"
        / "weights"
        / "lightning"
        / "model.ckpt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")

    return str(ckpt_path)


def build_model():
    return Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
        coreset_sampling_ratio=0.01,
    )


def load_patchcore_checkpoint(models_root: str, type_name: str):
    ckpt_path = get_ckpt_path(models_root, type_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_errors = []

    try:
        model = Patchcore.load_from_checkpoint(ckpt_path)
        model = model.to(device)
        model.eval()
        return model, device, ckpt_path
    except Exception as e:
        load_errors.append(f"load_from_checkpoint failed: {repr(e)}")

    try:
        model = build_model()
        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        return model, device, ckpt_path
    except Exception as e:
        load_errors.append(f"manual load_state_dict failed: {repr(e)}")

    raise RuntimeError(
        f"Checkpoint yüklenemedi: {ckpt_path}\n" + "\n".join(load_errors)
    )


def load_pil_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def pil_to_tensor(img: Image.Image, image_size=IMAGE_SIZE):
    img = img.resize(image_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def _extract_score(output):
    score = None

    if isinstance(output, dict):
        if "pred_scores" in output:
            val = output["pred_scores"]
            score = float(val[0].detach().cpu().item() if hasattr(val, "detach") else val[0])
        elif "pred_score" in output:
            val = output["pred_score"]
            score = float(val.detach().cpu().item() if hasattr(val, "detach") else val)
    else:
        if hasattr(output, "pred_scores"):
            val = output.pred_scores
            score = float(val[0].detach().cpu().item() if hasattr(val, "detach") else val[0])
        elif hasattr(output, "pred_score"):
            val = output.pred_score
            score = float(val.detach().cpu().item() if hasattr(val, "detach") else val)

    if score is None:
        raise RuntimeError(f"Score çıkarılamadı. Output type: {type(output)}, output: {output}")

    return score


def get_score(model, img_tensor, device):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    return _extract_score(output)


def predict_image(
    patchcore_model,
    device,
    pil_image: Image.Image,
    threshold: float,
    type_name: str,
):
    """
    PatchCore sadece good / defect kararını verir.
    Anomaly heatmap artık app.py tarafında Final Hybrid sistem ile üretilir.
    """
    tensor = pil_to_tensor(pil_image, IMAGE_SIZE)
    score = get_score(patchcore_model, tensor, device)
    pred = "defect" if score >= threshold else "good"

    return {
        "prediction": pred,
        "score": score,

        # Heatmap alanları Final Hybrid adapter tarafından defect durumda override edilir.
        "anomaly_map": None,
        "heatmap_overlay": None,
        "raw_heatmap_overlay": None,
        "marked_image": None,
        "binary_mask": None,
        "region_count": 0,
        "largest_bbox": None,
        "defect_crop": None,

        "heatmap_mode": "final_hybrid_pending" if pred == "defect" else "not_rendered_good",
        "display_heatmap_mode": "final_hybrid" if pred == "defect" else "-",

        "pixel_threshold": 0.0,
        "seed_threshold": 0.0,
        "mask_area_fraction": 0.0,
        "border_fraction": 0.0,
        "box_mode": "none",
        "ood_status": "ok",
        "guard_warnings": [],
    }