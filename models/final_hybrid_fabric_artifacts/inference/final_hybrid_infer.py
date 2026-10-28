
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as e:
    raise RuntimeError("timm eksik. Kurulum: pip install timm") from e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================
# Utils
# ============================================================

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_rgb(image_path, size=256):
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype(np.uint8)


def robust_norm(x, mode):
    x = x.astype(np.float32)

    if mode == "minmax":
        lo, hi = float(x.min()), float(x.max())
    elif mode == "q50_q999":
        lo, hi = float(np.quantile(x, 0.50)), float(np.quantile(x, 0.999))
    elif mode == "q70_q999":
        lo, hi = float(np.quantile(x, 0.70)), float(np.quantile(x, 0.999))
    elif mode == "q80_q999":
        lo, hi = float(np.quantile(x, 0.80)), float(np.quantile(x, 0.999))
    elif mode == "q90_q999":
        lo, hi = float(np.quantile(x, 0.90)), float(np.quantile(x, 0.999))
    elif mode == "q95_q999":
        lo, hi = float(np.quantile(x, 0.95)), float(np.quantile(x, 0.999))
    elif mode == "q97_q999":
        lo, hi = float(np.quantile(x, 0.97)), float(np.quantile(x, 0.999))
    else:
        raise ValueError(f"Unknown norm mode: {mode}")

    return np.clip((x - lo) / max(1e-8, hi - lo), 0, 1)


def component_filter(mask, heatmap, min_area=1, max_comp_area=0, keep_top_k=0, max_total_area=0, morph_open=0, morph_close=0):
    mask = mask.astype(np.uint8)

    if morph_open and morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(morph_open), int(morph_open)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if morph_close and morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(morph_close), int(morph_close)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comps.append((i, area))

    comps = sorted(comps, key=lambda x: x[1], reverse=True)

    if keep_top_k and int(keep_top_k) > 0:
        comps = comps[:int(keep_top_k)]

    out = np.zeros_like(mask)

    for comp_id, area in comps:
        comp_mask = labels == comp_id

        if max_comp_area and int(max_comp_area) > 0 and area > int(max_comp_area):
            coords = np.where(comp_mask)
            vals = heatmap[coords]
            order = np.argsort(vals)[::-1]
            keep = order[:int(max_comp_area)]
            out[coords[0][keep], coords[1][keep]] = 1
        else:
            out[comp_mask] = 1

    if max_total_area and int(max_total_area) > 0 and int(out.sum()) > int(max_total_area):
        coords = np.where(out > 0)
        vals = heatmap[coords]
        order = np.argsort(vals)[::-1]
        keep = order[:int(max_total_area)]

        capped = np.zeros_like(out)
        capped[coords[0][keep], coords[1][keep]] = 1
        out = capped

    return out.astype(np.uint8)


def colorize_heatmap(hm01):
    hm = np.clip(hm01 * 255, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return cm


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.45):
    out = rgb.copy().astype(np.float32)
    m = mask.astype(bool)

    if m.sum() > 0:
        c = np.zeros_like(out)
        c[..., 0] = color[0]
        c[..., 1] = color[1]
        c[..., 2] = color[2]
        out[m] = out[m] * (1 - alpha) + c[m] * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def save_outputs(out_dir, image_path, type_name, branch, rgb, heatmap01, pred_mask, extra=None):
    out_dir = ensure_dir(out_dir)

    original_path = out_dir / "original.png"
    heatmap_path = out_dir / "heatmap_color.png"
    mask_path = out_dir / "pred_mask.png"
    overlay_path = out_dir / "pred_overlay.png"
    result_path = out_dir / "result.json"

    Image.fromarray(rgb).save(original_path)
    Image.fromarray(colorize_heatmap(heatmap01)).save(heatmap_path)
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
    Image.fromarray(overlay_mask(rgb, pred_mask)).save(overlay_path)

    anomaly_score = float(heatmap01.max())
    pred_pixels = int(pred_mask.sum())
    pred_any = bool(pred_pixels > 0)

    result = {
        "image_path": str(image_path),
        "type_name": type_name,
        "branch": branch,
        "anomaly_score": anomaly_score,
        "pred_any": pred_any,
        "pred_pixels": pred_pixels,
        "outputs": {
            "original": str(original_path),
            "heatmap_color": str(heatmap_path),
            "pred_mask": str(mask_path),
            "pred_overlay": str(overlay_path),
        },
        "extra": extra or {},
    }

    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


# ============================================================
# DINO Extractor
# ============================================================

class DinoExtractor(torch.nn.Module):
    def __init__(self, model_name="vit_small_patch14_dinov2", img_size=448, patch_size=14, local_weight_path=None):
        super().__init__()
        self.model_name = model_name
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid = self.img_size // self.patch_size

        local_weight_path = Path(local_weight_path) if local_weight_path else None
        use_local = local_weight_path is not None and local_weight_path.exists()

        if use_local:
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    img_size=self.img_size,
                    dynamic_img_size=True,
                )
            except TypeError:
                self.model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    img_size=self.img_size,
                )

            ckpt = torch.load(local_weight_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(state, strict=False)

            if missing:
                print("DINO local weight missing keys:", len(missing))
            if unexpected:
                print("DINO local weight unexpected keys:", len(unexpected))

            print(f"DINO loaded from local weights: {local_weight_path}")

        else:
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.img_size,
                    dynamic_img_size=True,
                )
            except TypeError:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.img_size,
                )

            print("DINO loaded with pretrained=True via timm/HF cache.")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward_features_tokens(self, x):
        out = self.model.forward_features(x)

        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                tok = out["x_norm_patchtokens"]
            elif "x" in out:
                tok = out["x"]
            else:
                tok = out[list(out.keys())[0]]
        elif isinstance(out, (tuple, list)):
            tok = out[-1]
        else:
            tok = out

        if tok.ndim == 4:
            b, c, h, w = tok.shape
            tok = tok.flatten(2).transpose(1, 2).contiguous()
            return tok, h, w

        if tok.ndim != 3:
            raise ValueError(f"Unexpected token shape: {tok.shape}")

        b, n, c = tok.shape
        expected = self.grid * self.grid

        if n >= expected:
            patch_tok = tok[:, -expected:, :]
            h = w = self.grid
        else:
            side = int(np.sqrt(n))
            patch_tok = tok[:, :side * side, :]
            h = w = side

        return patch_tok, h, w

    def forward(self, x):
        with torch.no_grad():
            tok, h, w = self.forward_features_tokens(x)
            tok = F.normalize(tok, dim=-1)
        return tok, h, w


def preprocess_for_dino(image_path, dino_size):
    img = Image.open(image_path).convert("RGB").resize((int(dino_size), int(dino_size)), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    x = torch.from_numpy(arr).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    x = (x - mean) / std
    return x.unsqueeze(0)


def dino_score_map(extractor, image_path, bank_obj, device, mask_size=256, chunk_size=4096):
    x = preprocess_for_dino(image_path, extractor.img_size).to(device)

    with torch.no_grad():
        tokens, h, w = extractor(x)

    tokens = tokens[0].float().to(device)
    bank = bank_obj["bank"]

    best_sim = torch.full((tokens.shape[0],), -1e9, device=device)

    for start in range(0, bank.shape[0], int(chunk_size)):
        b = bank[start:start + int(chunk_size)].float().to(device)
        sim = tokens @ b.t()
        best_sim = torch.maximum(best_sim, sim.max(dim=1).values)

    dist = 1.0 - best_sim
    m = dist.view(1, 1, h, w)
    m = F.interpolate(m, size=(int(mask_size), int(mask_size)), mode="bilinear", align_corners=False)
    return m[0, 0].detach().cpu().numpy().astype(np.float32)


# ============================================================
# Type6 classical branch
# ============================================================

def to_gray_clahe(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def type6_classical_map(rgb, map_name):
    clahe = to_gray_clahe(rgb)

    if map_name.startswith("tophat_k"):
        k = int(map_name.split("_k")[-1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    if map_name.startswith("blackhat_k"):
        k = int(map_name.split("_k")[-1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    if map_name.startswith("max_top_black_k"):
        k = int(map_name.split("_k")[-1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel)
        return np.maximum(tophat, blackhat).astype(np.float32)

    if map_name.startswith("abs_median_resid_k"):
        k = int(map_name.split("_k")[-1])
        med = cv2.medianBlur(clahe, k)
        return np.abs(clahe.astype(np.float32) - med.astype(np.float32))

    raise ValueError(f"Unsupported type6 map_name: {map_name}")


# ============================================================
# Type8 classical branch helpers
# ============================================================

def norm01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return np.clip((x - mn) / max(1e-8, mx - mn), 0, 1)


def line_kernel(length=15, orientation="d1"):
    k = np.zeros((length, length), dtype=np.uint8)

    if orientation == "h":
        k[length // 2, :] = 1
    elif orientation == "v":
        k[:, length // 2] = 1
    elif orientation == "d1":
        np.fill_diagonal(k, 1)
    elif orientation == "d2":
        np.fill_diagonal(np.fliplr(k), 1)
    else:
        raise ValueError(orientation)

    return k


def type8_classical_line_map(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    k = line_kernel(15, "d1")
    opened = cv2.morphologyEx(clahe, cv2.MORPH_OPEN, k)
    diff = np.abs(clahe.astype(np.float32) - opened.astype(np.float32))
    return diff.astype(np.float32)


def dilate(mask, k):
    if int(k) <= 1:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
    return cv2.dilate(mask.astype(np.uint8), kernel)


# ============================================================
# Type9 RD-lite
# ============================================================

class TokenAutoEncoder(nn.Module):
    def __init__(self, in_dim=384, bottleneck_dim=48):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, int(bottleneck_dim)),
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(bottleneck_dim), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        y = F.normalize(y, dim=-1)
        return y


def rd_lite_score_map(extractor, ae, image_path, device, mask_size=256, token_batch_size=8192):
    x = preprocess_for_dino(image_path, extractor.img_size).to(device)

    with torch.no_grad():
        tokens, h, w = extractor(x)

    tokens = F.normalize(tokens[0].float(), dim=-1)

    errs = []
    ae.eval()

    with torch.no_grad():
        for start in range(0, tokens.shape[0], int(token_batch_size)):
            xt = tokens[start:start + int(token_batch_size)].to(device)
            pred = ae(xt)

            mse = ((pred - xt) ** 2).mean(dim=-1)
            cos = 1.0 - F.cosine_similarity(pred, xt, dim=-1)
            err = mse + 0.25 * cos

            errs.append(err.detach().cpu())

    err = torch.cat(errs, dim=0)
    m = err.view(1, 1, h, w)
    m = F.interpolate(m, size=(int(mask_size), int(mask_size)), mode="bilinear", align_corners=False)
    return m[0, 0].numpy().astype(np.float32)


# ============================================================
# Branch runners
# ============================================================

class HybridInfer:
    def __init__(self, pack_root, device=None):
        self.pack_root = Path(pack_root)
        self.config_path = self.pack_root / "configs" / "final_hybrid_config.json"

        if not self.config_path.exists():
            raise FileNotFoundError(self.config_path)

        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.table = self.config["final_branch_table"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.extractor_cache = {}
        self.bank_cache = {}
        self.ae_cache = {}

    def get_extractor(self, model_name, dino_size):
        key = (model_name, int(dino_size))
        if key not in self.extractor_cache:
            local_weight_path = self.pack_root / "checkpoints" / f"{model_name}_state_dict.pt"
            self.extractor_cache[key] = DinoExtractor(
                model_name=model_name,
                img_size=int(dino_size),
                local_weight_path=local_weight_path,
            ).to(self.device)
            self.extractor_cache[key].eval()
        return self.extractor_cache[key]

    def get_bank(self, bank_file):
        p = self.pack_root / bank_file
        if not p.exists():
            raise FileNotFoundError(p)
        if str(p) not in self.bank_cache:
            self.bank_cache[str(p)] = torch.load(p, map_location="cpu")
        return self.bank_cache[str(p)]

    def get_autoencoder(self, checkpoint_file, bottleneck_dim=48):
        p = self.pack_root / checkpoint_file
        if not p.exists():
            raise FileNotFoundError(p)

        key = (str(p), int(bottleneck_dim))
        if key not in self.ae_cache:
            ae = TokenAutoEncoder(in_dim=384, bottleneck_dim=int(bottleneck_dim)).to(self.device)
            state = torch.load(p, map_location=self.device)
            ae.load_state_dict(state)
            ae.eval()
            self.ae_cache[key] = ae

        return self.ae_cache[key]

    def run(self, image_path, type_name, out_dir):
        image_path = Path(image_path)
        out_dir = Path(out_dir)

        if type_name not in self.table:
            raise ValueError(f"Unknown type: {type_name}. Available: {list(self.table.keys())}")

        cfg = self.table[type_name]
        branch = cfg["branch"]
        mask_size = int(cfg.get("mask_size", self.config.get("mask_size", 256)))

        rgb = load_rgb(image_path, size=mask_size)

        if branch == "dino_feature_distance":
            result = self.run_dino_branch(image_path, type_name, cfg, rgb, out_dir)

        elif branch == "classical_small_defect":
            result = self.run_type6_classical(image_path, type_name, cfg, rgb, out_dir)

        elif branch == "dino_classical_fusion":
            result = self.run_type8_fusion(image_path, type_name, cfg, rgb, out_dir)

        elif branch == "dino_rd_lite":
            result = self.run_type9_rd_lite(image_path, type_name, cfg, rgb, out_dir)

        else:
            raise ValueError(f"Unsupported branch: {branch}")

        return result

    def run_dino_branch(self, image_path, type_name, cfg, rgb, out_dir):
        extractor = self.get_extractor(cfg["model_name"], cfg["dino_size"])
        bank_obj = self.get_bank(cfg["bank_file"])

        raw = dino_score_map(
            extractor,
            image_path,
            bank_obj,
            self.device,
            mask_size=int(cfg.get("mask_size", 256)),
        )

        hm = robust_norm(raw, cfg.get("norm_mode", cfg.get("mode", "minmax")))
        raw_pred = (hm >= float(cfg.get("threshold", 0.70))).astype(np.uint8)

        pred = component_filter(
            raw_pred,
            hm,
            min_area=int(cfg.get("min_area", 30)),
            max_comp_area=int(cfg.get("max_comp_area", 0)),
            keep_top_k=int(cfg.get("keep_top_k", 0)),
            max_total_area=int(cfg.get("max_total_area", 0)),
            morph_open=int(cfg.get("morph_open", 0)),
            morph_close=int(cfg.get("morph_close", 0)),
        )

        return save_outputs(out_dir, image_path, type_name, cfg["branch"], rgb, hm, pred, extra={"config": cfg})

    def run_type6_classical(self, image_path, type_name, cfg, rgb, out_dir):
        raw = type6_classical_map(rgb, cfg.get("map_name", "tophat_k9"))
        hm = robust_norm(raw, cfg.get("norm_mode", "q95_q999"))

        raw_pred = (hm >= float(cfg.get("threshold", 0.85))).astype(np.uint8)

        pred = component_filter(
            raw_pred,
            hm,
            min_area=int(cfg.get("min_area", 10)),
            max_comp_area=int(cfg.get("max_comp_area", 0)),
            keep_top_k=int(cfg.get("keep_top_k", 0)),
            max_total_area=int(cfg.get("max_total_area", 0)),
            morph_open=int(cfg.get("morph_open", 0)),
            morph_close=int(cfg.get("morph_close", 0)),
        )

        return save_outputs(out_dir, image_path, type_name, cfg["branch"], rgb, hm, pred, extra={"config": cfg})

    def run_type8_fusion(self, image_path, type_name, cfg, rgb, out_dir):
        extractor = self.get_extractor(cfg["model_name"], cfg["dino_size"])
        bank_obj = self.get_bank(cfg["bank_file"])

        dino_raw = dino_score_map(
            extractor,
            image_path,
            bank_obj,
            self.device,
            mask_size=int(cfg.get("mask_size", 256)),
        )

        class_raw = type8_classical_line_map(rgb)

        dino_hm = robust_norm(dino_raw, cfg.get("dino_norm", "minmax"))
        class_hm = robust_norm(class_raw, cfg.get("class_norm", "q95_q999"))

        mode = cfg.get("fusion_mode", "dino_gated_by_class")

        if mode == "weighted":
            alpha = float(cfg.get("alpha", 0.5))
            fused_hm = alpha * dino_hm + (1 - alpha) * class_hm
            raw_pred = (fused_hm >= float(cfg.get("weighted_threshold", 0.60))).astype(np.uint8)
            heat_for_filter = fused_hm

        else:
            dino_mask = (dino_hm >= float(cfg.get("dino_threshold", 0.55))).astype(np.uint8)
            class_mask = (class_hm >= float(cfg.get("class_threshold", 0.70))).astype(np.uint8)

            if mode == "union":
                raw_pred = ((dino_mask > 0) | (class_mask > 0)).astype(np.uint8)
                heat_for_filter = np.maximum(dino_hm, class_hm)

            elif mode == "intersection":
                raw_pred = ((dino_mask > 0) & (class_mask > 0)).astype(np.uint8)
                heat_for_filter = np.maximum(dino_hm, class_hm)

            elif mode == "class_gated_by_dino":
                gate = dilate(dino_mask, int(cfg.get("dilate_k", 3)))
                raw_pred = ((class_mask > 0) & (gate > 0)).astype(np.uint8)
                heat_for_filter = class_hm

            elif mode == "dino_gated_by_class":
                gate = dilate(class_mask, int(cfg.get("dilate_k", 3)))
                raw_pred = ((dino_mask > 0) & (gate > 0)).astype(np.uint8)
                heat_for_filter = dino_hm

            else:
                raise ValueError(f"Unsupported fusion_mode: {mode}")

        pred = component_filter(
            raw_pred,
            heat_for_filter,
            min_area=int(cfg.get("min_area", 20)),
            max_comp_area=int(cfg.get("max_comp_area", 0)),
            keep_top_k=int(cfg.get("keep_top_k", 0)),
            max_total_area=int(cfg.get("max_total_area", 0)),
            morph_open=int(cfg.get("morph_open", 0)),
            morph_close=int(cfg.get("morph_close", 0)),
        )

        return save_outputs(out_dir, image_path, type_name, cfg["branch"], rgb, robust_norm(heat_for_filter, "minmax"), pred, extra={"config": cfg})

    def run_type9_rd_lite(self, image_path, type_name, cfg, rgb, out_dir):
        extractor = self.get_extractor(cfg["model_name"], cfg["dino_size"])
        ae = self.get_autoencoder(cfg["checkpoint_file"], cfg.get("bottleneck_dim", 48))

        raw = rd_lite_score_map(
            extractor,
            ae,
            image_path,
            self.device,
            mask_size=int(cfg.get("mask_size", 256)),
        )

        hm = robust_norm(raw, cfg.get("mode", "minmax"))
        raw_pred = (hm >= float(cfg.get("threshold", 0.90))).astype(np.uint8)

        pred = component_filter(
            raw_pred,
            hm,
            min_area=int(cfg.get("min_area", 80)),
            max_comp_area=int(cfg.get("max_comp_area", 0)),
            keep_top_k=int(cfg.get("keep_top_k", 0)),
            max_total_area=int(cfg.get("max_total_area", 0)),
            morph_open=int(cfg.get("morph_open", 3)),
            morph_close=int(cfg.get("morph_close", 0)),
        )

        return save_outputs(out_dir, image_path, type_name, cfg["branch"], rgb, hm, pred, extra={"config": cfg})


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pack-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--type", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()

    infer = HybridInfer(args.pack_root, device=args.device)
    result = infer.run(args.image, args.type, args.out_dir)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
