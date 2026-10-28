import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


IMAGE_SIZE = (256, 256)


def infer_backbone_from_state_dict(state_dict: dict) -> str:
    if any(k.startswith("layer1.0.conv3") for k in state_dict.keys()):
        return "resnet50"

    fc_w = state_dict.get("fc.weight", None)
    if fc_w is not None:
        if fc_w.shape[1] == 2048:
            return "resnet50"
        if fc_w.shape[1] == 512:
            return "resnet18"

    return "resnet18"


def build_backbone(backbone_name: str, num_classes: int):
    backbone_name = (backbone_name or "").lower()

    if backbone_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")


def load_defect_type_classifier(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt["idx_to_class"]

    if isinstance(idx_to_class, dict):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    state_dict = ckpt["model_state_dict"]
    num_classes = len(class_to_idx)

    backbone = infer_backbone_from_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_backbone(backbone, num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device, class_to_idx, idx_to_class, backbone


def build_defect_type_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def predict_defect_type(model, device, pil_image: Image.Image, idx_to_class: dict):
    tf = build_defect_type_transform()
    x = tf(pil_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred_idx].item())

    pred_class = idx_to_class[pred_idx]
    return pred_class, conf


def pretty_defect_label(label: str) -> str:
    mapping = {
        "delik": "Delik",
        "yirtik": "Yırtık",
        "kumas_hatasi": "Kumaş Hatası",
        "leke_hatasi": "Leke Hatası",
        "diger": "Diğer",
    }
    return mapping.get(label, label.replace("_", " ").title())