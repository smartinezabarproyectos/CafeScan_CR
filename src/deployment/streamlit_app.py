"""CaféScan — Streamlit web app for coffee leaf disease detection.

Usage:
    streamlit run src/deployment/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.core.registry import build_model, list_models
from src.data.label_mapper import UNIFIED_LABELS
from src.deployment.inference import Predictor
from src.utils.io import load_checkpoint

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_DIR   = Path("results/checkpoints")
RESULTS_DIR = Path("results")
BEST_MODEL = "vit"  # recommended default

MODEL_METRICS = {
    "vit":             {"name": "ViT-Small",        "macro_f1": 0.9965, "accuracy": 0.9971, "params": "22M",  "badge": "⭐ Recomendado"},
    "efficientnet_b0": {"name": "EfficientNet-B0",  "macro_f1": 0.9540, "accuracy": 0.9645, "params": "5.3M", "badge": "⚡ Eficiente"},
    "mobilenet":       {"name": "MobileNetV3-Large","macro_f1": 0.9138, "accuracy": 0.9334, "params": "5.4M", "badge": "📱 Movil"},
    "resnet50":        {"name": "ResNet-50",         "macro_f1": 0.8039, "accuracy": 0.8598, "params": "25.6M","badge": "🔬 Clasico"},
}

CLASS_INFO = {
    "healthy": {
        "label": "Hoja Sana",
        "color": "#1e6b3c",
        "bg": "#d5eadc",
        "icon": "✅",
        "description": "La hoja no presenta signos de enfermedad. El cultivo se encuentra en buen estado fitosanitario.",
        "recommendation": "Continuar con el plan de fertilizacion y riego habitual. Realizar monitoreos periodicos.",
    },
    "rust": {
        "label": "Roya (Hemileia vastatrix)",
        "color": "#c55a11",
        "bg": "#fce4d6",
        "icon": "🟠",
        "description": "Manchas amarillo-anaranjadas en el enves de la hoja. Causa severas defoliaciones y reduccion de cosecha.",
        "recommendation": "Aplicar fungicidas triazoles o estrobulinas. Podar ramas afectadas. Mejorar ventilacion del cultivo.",
    },
    "cercospora": {
        "label": "Ojo de Gallo (Cercospora coffeicola)",
        "color": "#7030a0",
        "bg": "#e8d5f5",
        "icon": "🟣",
        "description": "Lesiones circulares con centro gris y halo amarillo. Afecta hojas, frutos y ramillas jovenes.",
        "recommendation": "Aplicar fungicidas cuprosos o mancozeb. Aumentar drenaje del suelo. Evitar exceso de sombreo.",
    },
    "miner": {
        "label": "Minador de la Hoja (Leucoptera coffeella)",
        "color": "#2e75b6",
        "bg": "#d6e4f0",
        "icon": "🔵",
        "description": "Galerias serpenteantes translucidas causadas por la larva del insecto minador. Reduce area fotosintetica.",
        "recommendation": "Aplicar insecticidas sistemicos (imidacloprid). Monitorear con trampas adhesivas. Control biologico con parasitoides.",
    },
    "phoma": {
        "label": "Phoma (Phoma costaricensis)",
        "color": "#c00000",
        "bg": "#ffd7d7",
        "icon": "🔴",
        "description": "Lesiones necroticas marron oscuro con halo clorotico. Frecuente en zonas de alta altitud con lluvias intensas.",
        "recommendation": "Aplicar fungicidas preventivos en epoca lluviosa. Eliminar tejido necrotico. Mejorar drenaje.",
    },
}


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor(model_name: str) -> Predictor:
    model = build_model(model_name, num_classes=5, pretrained=False)
    ckpt  = CKPT_DIR / f"{model_name}_best.pt"
    return Predictor(model, ckpt)


def available_models() -> list[str]:
    order = ["vit", "efficientnet_b0", "mobilenet", "resnet50"]
    return [m for m in order if (CKPT_DIR / f"{m}_best.pt").exists()]


# ---------------------------------------------------------------------------
# Grad-CAM (inline, no external deps beyond cv2)
# ---------------------------------------------------------------------------
def _get_gradcam_layer(model, model_name: str):
    if model_name == "efficientnet_b0":
        blocks = list(model.backbone.children())
        for layer in reversed(blocks):
            if any(isinstance(m, torch.nn.Conv2d) for m in layer.modules()):
                return layer
        return blocks[-1]
    elif model_name == "mobilenet":
        return model.backbone.blocks[-1]
    elif model_name == "resnet50":
        return model.backbone.layer4
    return None


def run_gradcam(model_name: str, pil_image: Image.Image, class_idx: int) -> np.ndarray | None:
    try:
        import cv2
        from src.data.transforms import test_transforms as tf
        predictor = load_predictor(model_name)
        model     = predictor.model
        layer     = _get_gradcam_layer(model, model_name)
        if layer is None:
            return None

        activations, gradients = {}, {}

        def fwd_hook(m, inp, out):
            activations["val"] = out.detach().clone()

        def bwd_hook(m, gin, gout):
            gradients["val"] = gout[0].detach().clone()

        h1 = layer.register_forward_hook(fwd_hook)
        h2 = layer.register_full_backward_hook(bwd_hook)

        transform = tf(224)
        x = transform(pil_image.convert("RGB")).unsqueeze(0).to(predictor.device)
        x = x.clone().requires_grad_(True)

        model.eval()
        with torch.enable_grad():
            logits = model(x)
            model.zero_grad()
            logits[0, class_idx].backward()

        h1.remove(); h2.remove()

        grads = gradients["val"].mean(dim=(2, 3), keepdim=True)
        cam   = (grads * activations["val"]).sum(dim=1).squeeze()
        cam   = torch.relu(cam).cpu().numpy()
        cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        img = np.array(pil_image.convert("RGB").resize((224, 224))).astype(float) / 255.0
        heatmap = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.55 * img + 0.45 * heatmap, 0, 1)
        return (overlay * 255).astype(np.uint8)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def class_badge(class_name: str) -> None:
    info = CLASS_INFO[class_name]
    st.markdown(
        f"""<div style="background:{info['bg']};border-left:6px solid {info['color']};
        padding:16px 20px;border-radius:8px;margin:12px 0">
        <span style="font-size:2rem">{info['icon']}</span>
        <span style="font-size:1.4rem;font-weight:700;color:{info['color']};margin-left:12px">
        {info['label']}</span></div>""",
        unsafe_allow_html=True,
    )


def prob_bars(probabilities: dict, pred_class: str) -> None:
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        info  = CLASS_INFO[cls]
        pct   = prob * 100
        bold  = "font-weight:700;" if cls == pred_class else ""
        color = info["color"] if cls == pred_class else "#888888"
        bar_w = max(int(pct), 1)
        st.markdown(
            f"""<div style="margin:4px 0">
            <span style="font-size:.85rem;{bold}color:{color};width:180px;display:inline-block">
            {info['icon']} {info['label']}</span>
            <div style="display:inline-block;width:{bar_w}%;max-width:55%;height:14px;
            background:{color};border-radius:4px;vertical-align:middle;margin:0 8px"></div>
            <span style="font-size:.85rem;{bold}color:{color}">{pct:.1f}%</span></div>""",
            unsafe_allow_html=True,
        )


def model_card(model_name: str) -> None:
    m = MODEL_METRICS.get(model_name, {})
    st.markdown(
        f"""<div style="background:#f0f4fa;border-radius:8px;padding:12px 16px;margin:6px 0">
        <b>{m.get('name','')}</b> &nbsp; <span style="color:#888;font-size:.8rem">{m.get('badge','')}</span><br>
        <span style="font-size:.85rem">Macro-F1: <b>{m.get('macro_f1',0)*100:.1f}%</b> &nbsp;|&nbsp;
        Accuracy: <b>{m.get('accuracy',0)*100:.1f}%</b> &nbsp;|&nbsp;
        Params: <b>{m.get('params','')}</b></span></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CaféScan — Deteccion de Enfermedades en Cafe",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Sidebar --
with st.sidebar:
    st.markdown("## ☕ CaféScan")
    st.caption("Deteccion de enfermedades en hojas de cafe mediante Deep Learning")
    st.divider()

    avail = available_models()
    if not avail:
        st.error("No hay checkpoints disponibles. Entrena al menos un modelo primero.")
        st.stop()

    default_idx = avail.index(BEST_MODEL) if BEST_MODEL in avail else 0
    model_name  = st.selectbox(
        "Modelo",
        avail,
        index=default_idx,
        format_func=lambda m: f"{MODEL_METRICS[m]['badge']}  {MODEL_METRICS[m]['name']}",
    )

    st.markdown("**Rendimiento en test set:**")
    model_card(model_name)

    st.divider()
    compare_mode = st.toggle("Modo comparacion (todos los modelos)", value=False)

    if model_name != "vit":
        gradcam_on = st.toggle("Mostrar Grad-CAM", value=False)
    else:
        gradcam_on = False
        st.caption("ℹ️ Grad-CAM no aplica a ViT (basado en atencion).")

    st.divider()
    st.markdown("**Clases detectadas:**")
    for cls, info in CLASS_INFO.items():
        st.caption(f"{info['icon']} {info['label']}")

# -- Header --
st.markdown("# ☕ CaféScan")
st.markdown("**Deteccion de enfermedades en hojas de cafe** — sube una imagen para clasificarla.")
st.divider()

# -- Upload --
col_up, col_cam = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Subir imagen de hoja de cafe",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
with col_cam:
    camera   = st.camera_input("Camara")

pil_image = None
if uploaded:
    pil_image = Image.open(uploaded).convert("RGB")
elif camera:
    pil_image = Image.open(camera).convert("RGB")

# ---------------------------------------------------------------------------
# Single-model prediction
# ---------------------------------------------------------------------------
if pil_image and not compare_mode:
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(pil_image, caption="Imagen cargada", use_column_width=True)

    with col_res:
        with st.spinner(f"Clasificando con {MODEL_METRICS[model_name]['name']}..."):
            t0        = time.time()
            predictor = load_predictor(model_name)
            result    = predictor.predict(pil_image)
            elapsed   = time.time() - t0

        pred_class = result["class"]
        confidence = result["confidence"]
        info       = CLASS_INFO[pred_class]

        st.markdown(f"### Resultado ({elapsed*1000:.0f} ms)")
        class_badge(pred_class)

        st.metric("Confianza", f"{confidence*100:.1f}%")
        st.markdown("**Probabilidades por clase:**")
        prob_bars(result["probabilities"], pred_class)

        st.divider()
        st.markdown(f"**Descripcion:** {info['description']}")
        st.info(f"**Recomendacion:** {info['recommendation']}", icon="💡")

    # Grad-CAM
    if gradcam_on:
        st.divider()
        st.markdown("### Mapa de activacion Grad-CAM")
        st.caption("Las zonas rojas/amarillas son las regiones que mas influyeron en la decision del modelo.")
        with st.spinner("Calculando Grad-CAM..."):
            pred_idx = UNIFIED_LABELS.index(pred_class)
            overlay  = run_gradcam(model_name, pil_image, pred_idx)
        if overlay is not None:
            col1, col2 = st.columns(2)
            col1.image(pil_image.resize((224, 224)), caption="Original", use_column_width=True)
            col2.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
        else:
            st.warning("No se pudo calcular Grad-CAM para este modelo.")

# ---------------------------------------------------------------------------
# Comparison mode — all models on same image
# ---------------------------------------------------------------------------
if pil_image and compare_mode:
    st.markdown("### Comparacion de modelos")
    st.caption("Misma imagen clasificada por los cuatro modelos.")

    cols  = st.columns(len(avail))
    results_all = {}

    for col, mname in zip(cols, avail):
        with col:
            minfo = MODEL_METRICS[mname]
            st.markdown(f"**{minfo['badge']}**")
            st.markdown(f"*{minfo['name']}*")
            with st.spinner(""):
                t0  = time.time()
                res = load_predictor(mname).predict(pil_image)
                dt  = time.time() - t0
            results_all[mname] = res

            pred   = res["class"]
            conf   = res["confidence"]
            cinfo  = CLASS_INFO[pred]
            st.markdown(
                f"""<div style="background:{cinfo['bg']};border-left:4px solid {cinfo['color']};
                padding:10px;border-radius:6px;margin:8px 0;text-align:center">
                <div style="font-size:1.5rem">{cinfo['icon']}</div>
                <div style="font-weight:700;color:{cinfo['color']};font-size:.9rem">{cinfo['label']}</div>
                <div style="color:#555;font-size:.85rem">{conf*100:.1f}% confianza</div>
                <div style="color:#aaa;font-size:.75rem">{dt*1000:.0f} ms</div></div>""",
                unsafe_allow_html=True,
            )

    # Agreement check
    preds = [r["class"] for r in results_all.values()]
    if len(set(preds)) == 1:
        st.success(f"✅ Todos los modelos coinciden: **{CLASS_INFO[preds[0]]['label']}**")
    else:
        majority = max(set(preds), key=preds.count)
        st.warning(f"⚠️ Los modelos no coinciden. Mayoria: **{CLASS_INFO[majority]['label']}** ({preds.count(majority)}/{len(preds)}). Se recomienda confiar en ViT.")

# -- Empty state --
if not pil_image:
    st.markdown(
        """<div style="text-align:center;padding:60px 20px;color:#aaa">
        <div style="font-size:4rem">🍃</div>
        <div style="font-size:1.2rem;margin-top:12px">Sube una imagen de hoja de cafe para comenzar</div>
        <div style="font-size:.9rem;margin-top:8px">Formatos soportados: JPG, JPEG, PNG</div></div>""",
        unsafe_allow_html=True,
    )
