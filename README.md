# Fabric Defect Detection – Kumaş Hata Tespiti

Staj projesi. Endüstriyel kumaş görüntülerinde hataları otomatik olarak tespit eden, hatanın yerini ısı haritasıyla gösteren ve hata tipini sınıflandıran Streamlit uygulaması.

## Nasıl çalışır

1. **Kumaş tipi (subtype) tespiti** – ResNet-50 sınıflandırıcı yüklenen görselin 10 kumaş / kamera tipinden hangisine ait olduğunu bulur; güven düşükse dağılım dışı (OOD) uyarısı verir.
2. **Anomali tespiti** – her kumaş tipi için ayrı eğitilmiş PatchCore (WideResNet-50, anomalib) modeli görseli *good / defect* olarak sınıflandırır; eşikler `configs/final_thresholds_itd_wr50_256.json` içindedir.
3. **Hata konumu (hibrit ısı haritası)** – kumaş tipine göre seçilen yöntemle hatalı bölge maskelenir:
   - DINOv2 (ViT-S/14, 448 px) özellik uzaklığı + bellek bankası
   - küçük hatalar için klasik görüntü işleme
   - DINOv2 + klasik yöntem füzyonu
   - DINOv2 tabanlı hafif reverse-distillation otokodlayıcı
4. **Hata tipi sınıflandırma** – ResNet-50 ile *delik, yırtık, kumaş hatası, leke hatası, diğer*.

Tip başına seçilen yöntemler ve parametreler `models/final_hybrid_fabric_artifacts/configs/final_hybrid_config.json` dosyasındadır.

## Proje yapısı

| Dosya | Açıklama |
|---|---|
| `app.py` | Streamlit arayüzü ve karar akışı |
| `subtype_utils.py` | Kumaş tipi sınıflandırıcı |
| `patchcore_utils.py` | PatchCore modellerini yükleme ve tahmin |
| `final_hybrid_adapter.py` | Hibrit ısı haritası / maske üretimi |
| `defect_type_utils.py` | Hata tipi sınıflandırıcı |
| `models/final_hybrid_fabric_artifacts/inference/final_hybrid_infer.py` | Hibrit çıkarım motoru (komut satırından da çalışır) |

## Teknolojiler

Python, PyTorch, anomalib (PatchCore), timm (DINOv2, WideResNet-50, ResNet-50), OpenCV, scikit-learn, Streamlit

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt  # birebir sürümler için: requirements_local_lock.txt
streamlit run app.py
```

Veri seti ve eğitilmiş model ağırlıkları (toplam birkaç GB) boyut nedeniyle repoya eklenmemiştir. Uygulama, eksik model dosyalarını açılışta listeler; beklenen klasörler:

```
models/all_types_wr50_256_l23_runs/<tip>_wr50_256_l23/Patchcore/itd_<tip>/v0/weights/lightning/model.ckpt
models/subtype_classifier/best_subtype_resnet50_256_good_anomaly.pth
models/defect_type_classifier/best_defect_type_resnet50_256_v10_5class.pth
models/final_hybrid_fabric_artifacts/banks/<tip>_dino_bank.pt
models/final_hybrid_fabric_artifacts/checkpoints/type9cam2_token_autoencoder.pt
```
