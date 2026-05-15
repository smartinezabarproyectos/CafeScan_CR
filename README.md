# ☕ CaféScan — Coffee Leaf Disease Detection

> Deep learning system for automatic classification of coffee leaf diseases.
> Compares four state-of-the-art architectures on a multi-source dataset of ~60,000 images.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Results

| Model | Accuracy | Macro-F1 | Params |
|---|---|---|---|
| ⭐ ViT-Small *(recommended)* | **99.71%** | **99.65%** | 22M |
| EfficientNet-B0 | 96.45% | 95.40% | 5.3M |
| MobileNetV3-Large | 93.34% | 91.38% | 5.4M |
| ResNet-50 | 85.98% | 80.39% | 25.6M |

Evaluated on an independent test set of **8,993 images** never seen during training. No overfitting detected (val–test gap < 0.2% for all models).

---

## Detected Classes

| Class | Disease | Description |
|---|---|---|
| ✅ `healthy` | Healthy leaf | No visible disease signs |
| 🟠 `rust` | Coffee rust (*Hemileia vastatrix*) | Yellow-orange spots on leaf underside |
| 🟣 `cercospora` | Brown eye spot (*Cercospora coffeicola*) | Circular lesions with yellow halo |
| 🔵 `miner` | Leaf miner (*Leucoptera coffeella*) | Translucent serpentine galleries |
| 🔴 `phoma` | Phoma (*Phoma costaricensis*) | Dark necrotic lesions with chlorotic halo |

---

## Demo Web App (Streamlit)

### Requirements

- Python 3.10 or higher
- A trained model checkpoint in `results/checkpoints/` (see [Training](#training))
- GPU recommended but not required

### 1. Clone the repository

```bash
git clone https://github.com/smartinezabarproyectos/CafeScan_CR.git
cd CafeScan_CR
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model checkpoints

Place the trained checkpoints in `results/checkpoints/`:

```
results/checkpoints/
    vit_best.pt               ← ViT-Small (recommended)
    efficientnet_b0_best.pt
    mobilenet_best.pt
    resnet50_best.pt
```

> **Note:** Checkpoints are not included in this repository due to their size.
> Contact the repository owner to obtain them, or [train your own](#training).

### 5. Launch the app

```bash
streamlit run src/deployment/streamlit_app.py
```

The app will open automatically at **http://localhost:8501**

---

## Using the Web App

### Single prediction

1. Select a model from the sidebar (ViT-Small is selected by default — best accuracy)
2. Upload a coffee leaf image (JPG, PNG) or use your webcam
3. The app returns:
   - Predicted disease class with confidence score
   - Probability distribution across all 5 classes
   - Disease description and treatment recommendation

### Comparison mode

Toggle **"Comparison mode"** in the sidebar to run all 4 models on the same image simultaneously and see where they agree or disagree.

### Grad-CAM visualization

For CNN models (EfficientNet, ResNet, MobileNet), toggle **"Show Grad-CAM"** to see a heatmap highlighting the image regions that most influenced the prediction.

> ViT does not support Grad-CAM (attention-based architecture).

---

## REST API

```bash
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
```

**Endpoint:**

```bash
curl -X POST "http://localhost:8000/predict?model_name=vit" \
     -F "file=@leaf.jpg"
```

**Response:**

```json
{
  "class": "rust",
  "confidence": 0.9823,
  "probabilities": {
    "healthy": 0.0041,
    "rust": 0.9823,
    "cercospora": 0.0089,
    "miner": 0.0031,
    "phoma": 0.0016
  }
}
```

---

## Training

### Train a single model

```bash
python scripts/train.py --model vit
python scripts/train.py --model efficientnet_b0 --epochs 50 --lr 1e-4
python scripts/train.py --model mobilenet --batch_size 64
python scripts/train.py --model resnet50 --patience 10
```

### Train all models overnight

```bash
python scripts/train_overnight.py
```

### Hyperparameter optimization (Optuna)

```bash
python -m src.experiments.hpo --model efficientnet_b0 --n_trials 30
```

### Final evaluation on test set

```bash
python scripts/final_eval.py
```

Generates confusion matrices, Grad-CAM maps, comparison figures and an overfitting report in `results/`.

---

## Project Structure

```
CafeScan_CR/
├── src/
│   ├── core/          # Config, model registry
│   ├── data/          # Dataset loaders, splits, transforms
│   ├── models/        # EfficientNet, ResNet, ViT, MobileNet
│   ├── training/      # Training loop, loss, callbacks
│   ├── evaluation/    # Metrics, confusion matrix, Grad-CAM
│   ├── experiments/   # HPO with Optuna
│   └── deployment/    # Streamlit app, FastAPI, Predictor
├── scripts/
│   ├── train.py              # Train a single model
│   ├── final_eval.py         # Test set evaluation
│   ├── train_overnight.py    # Train all models in sequence
│   └── training_ui.py        # Streamlit training dashboard
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   ├── 02_baseline_classical.ipynb   # Training curves
│   ├── 04_final_comparison.ipynb     # Test set results
│   └── 05_interpretability.ipynb     # Grad-CAM & confusion matrices
├── results/
│   ├── checkpoints/   # Model weights (not versioned)
│   ├── figures/       # Comparison plots
│   ├── tables/        # test_summary.csv, val_vs_test.csv
│   └── <model>/       # Per-model metrics, confusion matrix, Grad-CAM
├── data/
│   └── raw/           # Datasets (not versioned — ~20 GB)
│       ├── bracol/
│       ├── jmuben/
│       └── jmuben2/
├── requirements.txt
└── pyproject.toml
```

---

## Datasets

| Dataset | Source | Labels | Classes |
|---|---|---|---|
| BRACOL | Federal University of Viçosa, Brazil | CSV file | rust, cercospora, phoma, healthy |
| JMuBEN | University of Costa Rica | Folder structure | rust, cercospora, phoma |
| JMuBEN2 | University of Costa Rica | Folder structure | healthy, miner |

**Total:** ~59,950 valid images after filtering corrupted files.  
**Split:** 70% train / 15% val / 15% test — stratified by (class × dataset source).

---

## Requirements

```
torch >= 2.0
torchvision
timm
streamlit
fastapi
uvicorn
pillow
numpy
pandas
matplotlib
scikit-learn
optuna
opencv-python
```

Full list in [`requirements.txt`](requirements.txt).

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
