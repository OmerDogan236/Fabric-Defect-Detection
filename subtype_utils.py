import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

IMAGE_SIZE = (256, 256)


def load_subtype_classifier(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt["idx_to_class"]

    if isinstance(idx_to_class, dict):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, device, class_to_idx, idx_to_class


def build_subtype_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def _predict_subtype_probs(model, device, pil_image: Image.Image):
    tf = build_subtype_transform()
    x = tf(pil_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()

    return probs


def predict_subtype_with_topk(
    model,
    device,
    pil_image: Image.Image,
    idx_to_class: dict,
    topk: int = 5,
):
    """
    Eski predict_subtype davranışını bozmadan ek debug bilgisi üretir.
    Dış domain / Google görsellerinde subtype classifier'ın tek sınıfa çöküp çökmediğini
    görmek için top-k, margin ve entropy döndürür.
    """
    probs = _predict_subtype_probs(model, device, pil_image)
    num_classes = int(probs.numel())
    k = max(1, min(int(topk), num_classes))

    vals, inds = torch.topk(probs, k=k)
    topk_list = []
    for idx, val in zip(inds.tolist(), vals.tolist()):
        topk_list.append({
            "class": idx_to_class[int(idx)],
            "prob": float(val),
        })

    pred_class = topk_list[0]["class"]
    conf = float(topk_list[0]["prob"])
    second_conf = float(topk_list[1]["prob"]) if len(topk_list) > 1 else 0.0
    margin = float(conf - second_conf)

    eps = 1e-12
    entropy = float(-(probs * torch.log(probs + eps)).sum().item())
    max_entropy = float(math.log(max(2, num_classes)))
    entropy_norm = float(entropy / max_entropy)

    return pred_class, conf, topk_list, margin, entropy_norm


def predict_subtype(model, device, pil_image: Image.Image, idx_to_class: dict):
    pred_class, conf, _topk, _margin, _entropy = predict_subtype_with_topk(
        model=model,
        device=device,
        pil_image=pil_image,
        idx_to_class=idx_to_class,
        topk=1,
    )
    return pred_class, conf
