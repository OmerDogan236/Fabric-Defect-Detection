import os
from pathlib import Path

import streamlit as st

from patchcore_utils import (
    load_thresholds_json,
    load_patchcore_checkpoint,
    load_pil_image,
    predict_image,
)

from subtype_utils import (
    load_subtype_classifier,
    predict_subtype_with_topk,
)
from defect_type_utils import (
    load_defect_type_classifier,
    predict_defect_type,
    pretty_defect_label,
)

from final_hybrid_adapter import (
    is_final_hybrid_ready,
    run_final_hybrid_heatmap,
)

st.set_page_config(page_title="Fabric Defect Detection", layout="wide")

st.title("Fabric Defect Detection - ITD")
st.write(
    "Görsel yükle. Sistem subtype'ı otomatik bulsun, ardından good / defect kararı ve gerekiyorsa "
    "hata tipini göstersin."
)

BASE_DIR = Path(__file__).resolve().parent

MODELS_ROOT = str(BASE_DIR / "models" / "all_types_wr50_256_l23_runs")
THRESHOLD_JSON_PATH = str(BASE_DIR / "configs" / "final_thresholds_itd_wr50_256.json")
SUBTYPE_MODEL_PATH = str(BASE_DIR / "models" / "subtype_classifier" / "best_subtype_resnet50_256_good_anomaly.pth")
DEFECT_TYPE_MODEL_PATH = str(BASE_DIR / "models" / "defect_type_classifier" / "best_defect_type_resnet50_256_v10_5class.pth")
FINAL_HYBRID_PACK_ROOT = BASE_DIR / "models" / "final_hybrid_fabric_artifacts"
USE_FINAL_HYBRID_HEATMAP = True

missing_items = []

if not os.path.exists(MODELS_ROOT):
    missing_items.append(f"Models klasörü yok: {MODELS_ROOT}")
if not os.path.exists(THRESHOLD_JSON_PATH):
    missing_items.append(f"Threshold json yok: {THRESHOLD_JSON_PATH}")
if not os.path.exists(SUBTYPE_MODEL_PATH):
    missing_items.append(f"Subtype classifier modeli yok: {SUBTYPE_MODEL_PATH}")
if not os.path.exists(DEFECT_TYPE_MODEL_PATH):
    missing_items.append(f"Defect type classifier modeli yok: {DEFECT_TYPE_MODEL_PATH}")
if USE_FINAL_HYBRID_HEATMAP and not is_final_hybrid_ready():
    missing_items.append(f"Final hybrid heatmap paketi eksik: {FINAL_HYBRID_PACK_ROOT}")

if missing_items:
    for item in missing_items:
        st.error(item)
    st.stop()

FINAL_THRESHOLDS = load_thresholds_json(THRESHOLD_JSON_PATH)

TYPE_ORDER = [
    "type1cam1", "type2cam2", "type3cam1", "type4cam2", "type5cam2",
    "type6cam2", "type7cam2", "type8cam1", "type9cam2", "type10cam2",
]
TYPE_OPTIONS = [t for t in TYPE_ORDER if t in FINAL_THRESHOLDS]
TYPE_OPTIONS += [t for t in FINAL_THRESHOLDS.keys() if t not in TYPE_OPTIONS]


@st.cache_resource(show_spinner=True)
def load_subtype_model_cached(model_path: str):
    return load_subtype_classifier(model_path)


@st.cache_resource(show_spinner=True)
def load_patchcore_model_cached(models_root: str, type_name: str):
    return load_patchcore_checkpoint(models_root, type_name)


@st.cache_resource(show_spinner=True)
def load_defect_type_model_cached(model_path: str):
    return load_defect_type_classifier(model_path)


def show_guard_warnings(result: dict, subtype_conf: float):
    if subtype_conf < 0.80:
        st.warning("Subtype confidence düşük. Bu görüntü subtype routing için zor olabilir.")
    if subtype_conf < 0.60:
        st.error("Subtype confidence çok düşük. Sonuç OOD / yanlış subtype olabilir.")

    for w in result.get("guard_warnings", []):
        if w == "score_mask_inconsistency":
            st.warning("Skor defect diyor ama maske zayıf. Sonuç kararsız olabilir.")
        elif w == "border_dominant_activation":
            st.warning("Aktivasyon büyük ölçüde kenara yığılmış. Crop / border etkisi olabilir.")
        elif w == "excessive_mask_area":
            st.warning("Mask alanı çok büyük. Lighting shift veya domain farkı olabilir.")
        elif w == "many_regions":
            st.warning("Çok parçalı bölge çıktı. Görüntüde gürültü veya domain farkı olabilir.")

    ood_status = result.get("ood_status", "ok")
    if ood_status == "warn":
        st.warning("OOD guard: Bu sonuç dikkatli yorumlanmalı.")
    elif ood_status == "uncertain":
        st.error("OOD guard: Bu örnek güvenilmez / unseen domain olabilir.")


st.subheader("1) Görsel yükle")
uploaded_file = st.file_uploader(
    "PNG/JPG görsel yükle",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is not None:
    image = load_pil_image(uploaded_file)
    st.image(image, caption="Yüklenen görsel", width=420)

    st.markdown("### Subtype routing ayarı")
    manual_subtype = st.checkbox(
        "Subtype'ı manuel seç / override et",
        value=False,
        help="Google / dış domain görsellerinde otomatik subtype yanlış route ederse bunu açıp doğru type'ı seç.",
    )
    manual_type = st.selectbox(
        "Manuel kullanılacak subtype",
        TYPE_OPTIONS,
        index=0,
        disabled=not manual_subtype,
    )
    show_subtype_debug = st.checkbox(
        "Subtype top-k debug göster",
        value=True,
        help="Otomatik subtype modelinin ilk 5 tahminini, margin değerini ve OOD şüphesini gösterir.",
    )

    if st.button("Tahmin et"):
        with st.spinner("Önce subtype tahmini yapılıyor..."):
            subtype_model, subtype_device, class_to_idx, idx_to_class = load_subtype_model_cached(SUBTYPE_MODEL_PATH)
            auto_predicted_type, subtype_conf, subtype_topk, subtype_margin, subtype_entropy = predict_subtype_with_topk(
                subtype_model,
                subtype_device,
                image,
                idx_to_class,
                topk=5,
            )

        predicted_type = manual_type if manual_subtype else auto_predicted_type
        selected_threshold = float(FINAL_THRESHOLDS[predicted_type])

        if show_subtype_debug:
            st.write(
                "Subtype top-k: " +
                " | ".join([f"{x['class']}: {x['prob']:.4f}" for x in subtype_topk])
            )
            st.write(
                f"Auto subtype: **{auto_predicted_type}** | Kullanılan subtype: **{predicted_type}** | "
                f"Top1 conf: **{subtype_conf:.4f}** | Top1-Top2 margin: **{subtype_margin:.4f}** | "
                f"Entropy norm: **{subtype_entropy:.4f}**"
            )

        if manual_subtype:
            st.warning(
                f"Manuel subtype override aktif. Otomatik model {auto_predicted_type} dedi, "
                f"ama analiz {predicted_type} üzerinden çalıştırılıyor."
            )
        elif subtype_conf < 0.80 or subtype_margin < 0.15:
            st.warning(
                "Subtype otomatik routing zayıf olabilir. Google/dış domain görsellerinde manuel subtype seçmeyi dene."
            )

        with st.spinner(f"{predicted_type} PatchCore modeli yükleniyor ve analiz yapılıyor..."):
            patchcore_model, patchcore_device, ckpt_path = load_patchcore_model_cached(
                MODELS_ROOT,
                predicted_type,
            )

            result = predict_image(
                patchcore_model,
                patchcore_device,
                image,
                selected_threshold,
                predicted_type,
            )

        pred = result["prediction"]
        score = result["score"]

        final_heatmap_warning = None

        if pred == "defect" and USE_FINAL_HYBRID_HEATMAP:
            with st.spinner(f"{predicted_type} Final Hybrid anomaly heatmap üretiliyor..."):
                try:
                    final_heatmap_fields = run_final_hybrid_heatmap(
                        pil_image=image,
                        type_name=predicted_type,
                        device=None,  # cuda varsa cuda, yoksa cpu
                    )
                    result.update(final_heatmap_fields)
                except Exception as e:
                    final_heatmap_warning = str(e)

        defect_type_pred = None
        defect_type_conf = None
        defect_type_input = None
        defect_type_pretty = None
        defect_backbone = None

        if pred == "defect":
            with st.spinner("Hata tipi tahmini yapılıyor..."):
                defect_type_model, defect_type_device, dt_class_to_idx, dt_idx_to_class, defect_backbone = load_defect_type_model_cached(DEFECT_TYPE_MODEL_PATH)

                defect_type_input = image

                defect_type_pred, defect_type_conf = predict_defect_type(
                    defect_type_model,
                    defect_type_device,
                    defect_type_input,
                    dt_idx_to_class,
                )
                defect_type_pretty = pretty_defect_label(defect_type_pred)

        metric_cols = st.columns(9)
        metric_cols[0].metric("Used Subtype", predicted_type)
        metric_cols[1].metric("Auto Subtype", auto_predicted_type)
        metric_cols[2].metric("Subtype Conf", f"{subtype_conf:.4f}")
        metric_cols[3].metric("Margin", f"{subtype_margin:.4f}")
        metric_cols[4].metric("Prediction", pred)
        metric_cols[5].metric("Score", f"{score:.6f}")
        metric_cols[6].metric("Threshold", f"{selected_threshold:.6f}")
        metric_cols[7].metric("Heatmap", result.get("heatmap_mode", "-"))
        metric_cols[8].metric("OOD", result.get("ood_status", "-"))

        st.caption(f"PatchCore checkpoint: {ckpt_path}")
        if final_heatmap_warning:
            st.warning(
                f"Final Hybrid heatmap üretilemedi. Good/defect ve hata tipi sonucu gösterildi, ancak heatmap görseli oluşturulamadı: {final_heatmap_warning}"
            )
        else:
            st.caption(f"Final hybrid heatmap: {result.get('heatmap_mode', '-')}")
            if result.get("final_hybrid_branch"):
                st.caption(f"Heatmap route/branch: {result.get('final_hybrid_branch')}")
            if result.get("final_hybrid_out_dir"):
                st.caption(f"Final hybrid output: {result.get('final_hybrid_out_dir')}")
        st.caption(f"Display heatmap mode: {result.get('display_heatmap_mode', '-')}")

        if pred == "defect":
            st.error(f"Sonuç: DEFECT - {defect_type_pretty if defect_type_pretty else '-'}")
            if defect_type_conf is not None:
                st.write(f"Hata tipi confidence: **{defect_type_conf:.4f}**")
            if defect_backbone is not None:
                st.caption(f"Defect type backbone: {defect_backbone}")
        else:
            st.success("Sonuç: GOOD")

        if pred == "defect" and defect_type_conf is not None and defect_type_conf < 0.60:
            st.warning("Hata tipi confidence düşük. Bu örnek sınıflar arasında karışıyor olabilir.")

        show_guard_warnings(result, subtype_conf)

        st.subheader("2) Görsel çıktıları")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(image, caption="Orijinal", width=360)

        if pred == "defect":
            with c2:
                if result["heatmap_overlay"] is not None:
                    st.image(result["heatmap_overlay"], caption="Anomaly Heatmap (Display)", width=360)

            with c3:
                if result["marked_image"] is not None:
                    st.image(result["marked_image"], caption="İşaretlenmiş Bölge", width=360)

            extra_cols = st.columns(3)
            if result["binary_mask"] is not None:
                extra_cols[0].image(
                    result["binary_mask"],
                    caption=f"Binary Mask (Region count: {result['region_count']})",
                    width=300,
                )

            if result.get("raw_heatmap_overlay") is not None:
                extra_cols[1].image(
                    result["raw_heatmap_overlay"],
                    caption="Raw Heatmap",
                    width=300,
                )

            if defect_type_input is not None:
                extra_cols[2].image(defect_type_input, caption="Defect Type Input", width=260)

            if result["largest_bbox"] is not None:
                x, y, w, h = result["largest_bbox"]
                st.write(f"Detected bbox: x={x}, y={y}, w={w}, h={h}")

            mask_area_fraction = result.get("mask_area_fraction", 0.0) or 0.0
            border_fraction = result.get("border_fraction", 0.0) or 0.0
            pixel_threshold = result.get("pixel_threshold", 0.0) or 0.0

            st.write(
                f"Mask area frac: **{mask_area_fraction:.4f}** | "
                f"Border frac: **{border_fraction:.4f}** | "
                f"Pixel thr: **{pixel_threshold:.4f}**"
            )
        else:
            with c2:
                st.info("Görsel good olarak değerlendirildi. Şüpheli bölge işaretlemesi gösterilmiyor.")

st.markdown("---")
st.write("Notlar:")
st.write("- Bu sürüm subtype'ı otomatik tahmin eder.")
st.write("- Good / defect kararı için ilgili subtype PatchCore modeli kullanılır.")
st.write("- Defect tespit edilirse anomaly heatmap, Final Hybrid sistemiyle üretilir.")
st.write("- Final Hybrid sistem; type6 için bright/contrast branch, diğer typelar için eski Final Hybrid branch kullanır.")
st.write("- Hata tipi sınıflandırması defect sonucundan sonra ayrıca çalıştırılır.")
