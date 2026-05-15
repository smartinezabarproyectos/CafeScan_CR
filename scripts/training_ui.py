"""QuantumCafe-CR Training Dashboard

Mini GUI for launching model training and Optuna HPO.

Usage:
    streamlit run scripts/training_ui.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.registry import list_models

PYTHON = str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

ALL_MODELS = list_models()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QuantumCafe-CR — Training Dashboard",
    page_icon="☕",
    layout="wide",
)

st.title("☕ QuantumCafe-CR — Training Dashboard")
st.caption("Coffee Leaf Disease Detection — DL Model Comparison")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Global Config")
    data_root = st.text_input("Data Root", "data/raw")
    results_dir = st.text_input("Results Dir", "results")
    seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, step=1)
    st.divider()
    st.markdown("**Models available**")
    st.markdown(", ".join(f"`{m}`" for m in ALL_MODELS))
    st.divider()
    st.caption("Logs stream live while training runs.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_proc(cmd: list[str], log_area, parse_hpo: bool = False) -> tuple[int, list[str], dict]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        bufsize=1,
    )

    lines: list[str] = []
    hpo_info: dict = {"trials": [], "best": None, "best_params": None}
    hpo_metric = st.empty() if parse_hpo else None

    for line in proc.stdout:
        lines.append(line)
        log_area.code("".join(lines[-120:]), language=None)

        if not parse_hpo:
            continue

        stripped = line.strip()
        if stripped.startswith("[TRIAL_DONE]"):
            parts = dict(p.split("=") for p in stripped.split()[1:])
            entry = {
                "trial": int(parts["trial"]),
                "value": float(parts["value"]),
                "best":  float(parts["best"]),
            }
            hpo_info["trials"].append(entry)
            hpo_info["best"] = entry["best"]
            hpo_metric.metric(
                label=f"Trial {entry['trial'] + 1} done",
                value=f"Best F1: {entry['best']:.4f}",
                delta=f"This trial: {entry['value']:.4f}",
            )
        elif stripped.startswith("[TRIAL_PRUNED]"):
            parts = dict(p.split("=") for p in stripped.split()[1:])
            hpo_metric.metric(
                label=f"Trial {parts['trial']} pruned (Hyperband)",
                value=f"Best so far: {hpo_info['best'] or '—'}",
            )
        elif stripped.startswith("[BEST_PARAMS]"):
            params_json = stripped[len("[BEST_PARAMS] "):]
            hpo_info["best_params"] = json.loads(params_json)

    proc.wait()
    return proc.returncode, lines, hpo_info


def _load_all_hpo_from_db(db_path_str: str) -> dict[str, dict]:
    results = {}
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        db_file = PROJECT_ROOT / db_path_str
        if not db_file.exists():
            return results
        storage = f"sqlite:///{db_file}"
        summaries = optuna.get_all_study_summaries(storage=storage)
        for summary in summaries:
            name = summary.study_name
            if not name.startswith("hpo_"):
                continue
            model = name[len("hpo_"):]
            try:
                study = optuna.load_study(study_name=name, storage=storage)
                completed = [t for t in study.trials if t.state.name == "COMPLETE"]
                if not completed:
                    continue
                running_best, trials_list = float("-inf"), []
                for t in sorted(completed, key=lambda x: x.number):
                    running_best = max(running_best, t.value)
                    trials_list.append({"trial": t.number, "value": t.value, "best": running_best})
                results[model] = {
                    "best": study.best_value,
                    "best_params": study.best_params,
                    "trials": trials_list,
                }
            except Exception:
                pass
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "hpo_results" not in st.session_state:
    st.session_state.hpo_results = {}

if "hpo_db_loaded" not in st.session_state:
    st.session_state.hpo_db_loaded = False

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_train, tab_hpo, tab_results, tab_all = st.tabs(
    ["🚂 Training", "🔬 HPO (Optuna)", "📊 Results", "🌙 Run All"]
)


# ===========================================================================
# Tab 1 — Standard Training
# ===========================================================================

with tab_train:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Select Models")
        selected = [m for m in ALL_MODELS if st.checkbox(m, value=True, key=f"sel_{m}")]

    with right:
        st.subheader("Training Config")
        col1, col2 = st.columns(2)
        with col1:
            t_epochs = st.slider("Epochs", 5, 100, 30, key="t_epochs")
            t_batch  = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=64, key="t_batch")
            t_lr = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                key="t_lr",
            )
        with col2:
            t_wd = st.select_slider(
                "Weight Decay",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                key="t_wd",
            )
            t_patience = st.slider("Early Stop Patience", 3, 20, 8, key="t_patience")
            t_ls = st.slider("Label Smoothing", 0.0, 0.3, 0.1, step=0.01, key="t_ls")

    st.divider()

    if not selected:
        st.warning("Select at least one model above.")

    if st.button("▶ Start Training", type="primary", disabled=not selected, key="btn_train"):
        log_area = st.empty()
        cmd = [
            PYTHON, "-m", "src.experiments.runner",
            "--models", *selected,
            "--epochs",          str(t_epochs),
            "--seed",            str(seed),
            "--data_root",       data_root,
            "--lr",              str(t_lr),
            "--weight_decay",    str(t_wd),
            "--batch_size",      str(t_batch),
            "--patience",        str(t_patience),
            "--label_smoothing", str(t_ls),
        ]
        st.info(f"Training: {', '.join(selected)}")
        rc, _, _ = _stream_proc(cmd, log_area)
        if rc == 0:
            st.success("✅ Training complete! Go to the **Results** tab.")
            st.balloons()
        else:
            st.error(f"❌ Training failed (exit code {rc}). Check logs above.")


# ===========================================================================
# Tab 2 — HPO (Optuna)
# ===========================================================================

with tab_hpo:
    st.info(
        "HPO with **Optuna** (TPE sampler + Hyperband pruner) finds the best hyperparameters "
        "for a model. Results are saved to SQLite and can be reloaded across sessions."
    )

    left, right = st.columns([1, 2])

    with left:
        st.subheader("HPO Settings")
        hpo_model   = st.selectbox("Model to optimize", ALL_MODELS, key="hpo_model")
        n_trials    = st.slider("Number of trials", 10, 100, 30, key="hpo_trials")
        hpo_epochs  = st.slider("Epochs per trial", 5, 30, 15, key="hpo_epochs")
        hpo_db      = st.text_input("Optuna DB (SQLite)", f"{results_dir}/optuna.db", key="hpo_db")
        st.caption(f"⏱ ~{n_trials} trials × {hpo_epochs} epochs. Hyperband prunes bad trials early.")

    with right:
        st.subheader("Search Space")
        st.markdown("""
| Parameter | Range | Method |
|---|---|---|
| `lr` | 1e-5 → 1e-2 | log-uniform |
| `weight_decay` | 1e-5 → 1e-2 | log-uniform |
| `batch_size` | 32, 64, 128 | categorical |
| `label_smoothing` | 0.0 → 0.2 | uniform |
| `backbone_lr_mult` | 0.01 → 0.5 | uniform |
| `warmup_epochs` | 1 → 5 | integer |
        """)

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_hpo_btn = st.button("🔬 Start HPO", type="primary", key="btn_hpo")
    with col_btn2:
        if st.button("📂 Cargar resultados anteriores del DB", key="btn_load_db"):
            db_file = PROJECT_ROOT / hpo_db
            if not db_file.exists():
                st.warning(f"No se encontro {hpo_db}. Corre el HPO primero.")
            else:
                try:
                    import optuna
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    study = optuna.load_study(
                        study_name=f"hpo_{hpo_model}",
                        storage=f"sqlite:///{db_file}",
                    )
                    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if completed:
                        trials_list = []
                        running_best = float("-inf")
                        for t in sorted(completed, key=lambda t: t.number):
                            running_best = max(running_best, t.value)
                            trials_list.append({"trial": t.number, "value": t.value, "best": running_best})
                        st.session_state.hpo_results[hpo_model] = {
                            "best": study.best_value,
                            "best_params": study.best_params,
                            "trials": trials_list,
                        }
                        st.success(f"Cargado: {len(completed)} trials. Best F1={study.best_value:.4f}")
                        st.rerun()
                    else:
                        st.warning("El estudio existe pero no tiene trials completados todavia.")
                except Exception as e:
                    st.error(f"No se pudo cargar el estudio 'hpo_{hpo_model}': {e}")

    if run_hpo_btn:
        cmd = [
            PYTHON, "-m", "src.experiments.hpo",
            "--model",      hpo_model,
            "--n_trials",   str(n_trials),
            "--hpo_epochs", str(hpo_epochs),
            "--data_root",  data_root,
            "--seed",       str(seed),
            "--db",         hpo_db,
        ]
        st.info(f"Optimizing **{hpo_model}** — {n_trials} trials × {hpo_epochs} epochs")
        log_area = st.empty()
        rc, _, hpo_info = _stream_proc(cmd, log_area, parse_hpo=True)

        if rc == 0 and hpo_info.get("best_params"):
            st.session_state.hpo_results[hpo_model] = hpo_info
            st.rerun()
        elif rc != 0:
            st.error(f"❌ HPO failed (exit {rc}). Check logs above.")

    saved = st.session_state.hpo_results.get(hpo_model)
    if saved and saved.get("best_params"):
        st.success(f"✅ Best val macro-F1 for **{hpo_model}**: **{saved['best']:.4f}**")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.subheader("Best Hyperparameters")
            params = saved["best_params"]
            st.dataframe(
                pd.DataFrame({"Parameter": list(params.keys()), "Best Value": list(params.values())}),
                use_container_width=True, hide_index=True,
            )
        with col_r2:
            if saved["trials"]:
                st.subheader("Trial History")
                trials_df = pd.DataFrame(saved["trials"]).set_index("trial")
                st.line_chart(trials_df[["value", "best"]])

        st.subheader("Train with Best Params")
        st.caption("HPO solo busca los mejores hiperparametros — NO guarda un modelo entrenado.")
        params = saved["best_params"]
        bs = params.get("batch_size", 64)

        if st.button(f"▶ Entrenar {hpo_model} con estos params", key="btn_train_best"):
            log_area2 = st.empty()
            cmd2 = [
                PYTHON, "-m", "src.experiments.runner",
                "--models",          hpo_model,
                "--epochs",          "30",
                "--seed",            str(seed),
                "--data_root",       data_root,
                "--batch_size",      str(bs),
                "--lr",              str(params.get("lr", 1e-4)),
                "--weight_decay",    str(params.get("weight_decay", 1e-4)),
                "--patience",        "8",
                "--label_smoothing", str(params.get("label_smoothing", 0.1)),
            ]
            st.info(f"Entrenando **{hpo_model}** con los mejores params de HPO...")
            rc2, _, _ = _stream_proc(cmd2, log_area2)
            if rc2 == 0:
                st.success("✅ Listo. Ve al tab Results.")
                st.balloons()
            else:
                st.error(f"❌ Fallo (exit {rc2}).")


# ===========================================================================
# Tab 3 — Results
# ===========================================================================

with tab_results:
    if st.button("🔄 Refresh", key="btn_refresh"):
        st.rerun()

    results_path = PROJECT_ROOT / results_dir
    rows = []
    if results_path.exists():
        for model_dir in sorted(results_path.iterdir()):
            if not model_dir.is_dir():
                continue
            metrics_file = model_dir / "val_metrics.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                m = json.load(f)
            rows.append({
                "model":           model_dir.name,
                "accuracy":        m.get("accuracy", 0),
                "macro_f1":        m.get("macro_f1", 0),
                "weighted_f1":     m.get("weighted_f1", 0),
                "macro_precision": m.get("macro_precision", 0),
                "macro_recall":    m.get("macro_recall", 0),
            })

    if not rows:
        st.warning("No results found yet. Run training first.")
    else:
        df = pd.DataFrame(rows).set_index("model").sort_values("macro_f1", ascending=False)

        st.subheader("Model Comparison — Validation Set")
        st.dataframe(
            df.style
              .highlight_max(axis=0, color="#d4edda")
              .highlight_min(axis=0, color="#f8d7da")
              .format("{:.4f}"),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Val Macro F1")
            st.bar_chart(df["macro_f1"])
        with col2:
            st.subheader("Val Accuracy")
            st.bar_chart(df["accuracy"])

        fig_dir = results_path / "figures"
        for fname, title in [
            ("model_comparison.png", "Full Metric Comparison"),
            ("per_class_f1.png",     "Per-Class F1 Scores"),
        ]:
            fig_path = fig_dir / fname
            if fig_path.exists():
                st.subheader(title)
                st.image(str(fig_path), use_column_width=True)


# ===========================================================================
# Tab 4 — Run All
# ===========================================================================

def _get_params_for(model: str, fallback_lr, fallback_wd, fallback_bs, fallback_ls) -> dict:
    saved = st.session_state.hpo_results.get(model)
    if saved and saved.get("best_params"):
        return {**saved["best_params"], "_source": f"HPO  F1={saved['best']:.4f}"}
    if model == "resnet50":
        saved_eff = st.session_state.hpo_results.get("efficientnet_b0")
        if saved_eff and saved_eff.get("best_params"):
            return {**saved_eff["best_params"], "_source": f"Hereda efficientnet_b0  F1={saved_eff['best']:.4f}"}
    return {
        "lr": fallback_lr, "weight_decay": fallback_wd,
        "batch_size": fallback_bs, "label_smoothing": fallback_ls,
        "_source": "default",
    }


with tab_all:
    st.subheader("Pipeline completo: HPO pendientes → Entrenar todo")

    hpo_db_path = f"{results_dir}/optuna.db"

    if not st.session_state.hpo_db_loaded:
        loaded = _load_all_hpo_from_db(hpo_db_path)
        for m, info in loaded.items():
            st.session_state.hpo_results[m] = info
        st.session_state.hpo_db_loaded = True

    col_reload, _ = st.columns([1, 3])
    with col_reload:
        if st.button("🔄 Recargar HPO del DB", key="btn_reload_db"):
            loaded = _load_all_hpo_from_db(hpo_db_path)
            for m, info in loaded.items():
                st.session_state.hpo_results[m] = info
            st.rerun()

    # ── Estado HPO ────────────────────────────────────────────────────────────
    st.markdown("### Estado del HPO por modelo")
    status_rows, pending_models = [], []
    for m in ALL_MODELS:
        saved = st.session_state.hpo_results.get(m)
        if saved and saved.get("best_params"):
            p = saved["best_params"]
            status_rows.append({
                "Modelo": m, "HPO": "✅ Listo", "Best F1": f"{saved['best']:.4f}",
                "lr": f"{p.get('lr', '—'):.2e}" if isinstance(p.get("lr"), float) else "—",
                "weight_decay": f"{p.get('weight_decay', '—'):.2e}" if isinstance(p.get("weight_decay"), float) else "—",
                "batch_size": p.get("batch_size", "—"),
                "label_smoothing": p.get("label_smoothing", "—"),
            })
        else:
            status_rows.append({
                "Modelo": m, "HPO": "❌ Pendiente", "Best F1": "—",
                "lr": "—", "weight_decay": "—", "batch_size": "—", "label_smoothing": "—",
            })
            pending_models.append(m)

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown("### HPO pendientes")
    if not pending_models:
        st.success("Todos los modelos tienen HPO completado.")
        hpo_to_run = []
    else:
        st.caption("Selecciona los modelos que aun necesitan HPO antes de entrenar:")
        cols = st.columns(len(pending_models))
        hpo_to_run = [
            m for m, col in zip(pending_models, cols)
            if col.checkbox(m, value=True, key=f"pending_hpo_{m}")
        ]

    if hpo_to_run:
        with st.expander("⚙️ Config HPO para modelos pendientes", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                all_n_trials  = st.slider("Trials por modelo", 10, 100, 50, key="all_n_trials")
            with c2:
                all_hpo_epochs = st.slider("Epochs por trial", 5, 30, 20, key="all_hpo_epochs")
    else:
        all_n_trials, all_hpo_epochs = 50, 20

    st.divider()

    # ── Config entrenamiento ──────────────────────────────────────────────────
    st.markdown("### Config entrenamiento")
    c1, c2, c3 = st.columns(3)
    with c1:
        all_epochs = st.slider("Epochs", 20, 100, 40, key="all_epochs")
    with c2:
        fallback_lr = st.select_slider("LR (fallback)", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                       value=1e-4, format_func=lambda x: f"{x:.0e}", key="all_lr")
        fallback_wd = st.select_slider("WD (fallback)", [1e-5, 1e-4, 1e-3],
                                       value=1e-4, format_func=lambda x: f"{x:.0e}", key="all_wd")
    with c3:
        fallback_bs = st.select_slider("Batch (fallback)", [32, 64, 128], value=64, key="all_bs")
        fallback_ls = st.slider("Label Smoothing (fallback)", 0.0, 0.2, 0.1, step=0.01, key="all_ls")

    # ── Estado de entrenamiento ───────────────────────────────────────────────
    st.markdown("### Estado del entrenamiento por modelo")

    def _training_status(model: str) -> tuple[str, str]:
        metrics_file = PROJECT_ROOT / results_dir / model / "val_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    m = json.load(f)
                return "✅ Entrenado", f"{m.get('macro_f1', 0):.4f}"
            except Exception:
                pass
        return "❌ Pendiente", "—"

    train_status_rows, pending_train = [], []
    for m in ALL_MODELS:
        estado, f1 = _training_status(m)
        p = _get_params_for(m, fallback_lr, fallback_wd, fallback_bs, fallback_ls)
        train_status_rows.append({
            "Modelo": m, "Entrenamiento": estado,
            "Val F1": f1, "HP Fuente": p.get("_source", "default"),
        })
        if estado == "❌ Pendiente":
            pending_train.append(m)

    st.dataframe(pd.DataFrame(train_status_rows), use_container_width=True, hide_index=True)

    st.markdown("### Modelos a entrenar")
    if not pending_train:
        st.success("Todos los modelos ya estan entrenados.")
        models_to_train = []
    else:
        cols = st.columns(max(len(ALL_MODELS), 1))
        models_to_train = []
        for m, col in zip(ALL_MODELS, cols):
            already_done = m not in pending_train
            label = f"{m} ({'ya listo' if already_done else 'pendiente'})"
            if col.checkbox(label, value=not already_done, key=f"train_sel_{m}"):
                models_to_train.append(m)

    with st.expander("👁️ Preview hiperparametros finales por modelo"):
        preview_rows = []
        for m in (models_to_train or ALL_MODELS):
            p = _get_params_for(m, fallback_lr, fallback_wd, fallback_bs, fallback_ls)
            preview_rows.append({
                "Modelo": m,
                "lr": p.get("lr", fallback_lr),
                "weight_decay": p.get("weight_decay", fallback_wd),
                "batch_size": p.get("batch_size", fallback_bs),
                "label_smoothing": p.get("label_smoothing", fallback_ls),
                "Fuente": p.get("_source", "default"),
            })
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    st.divider()

    parts = []
    if hpo_to_run:
        parts.append(f"HPO ({', '.join(hpo_to_run)})")
    if models_to_train:
        parts.append(f"Entrenar ({', '.join(models_to_train)})")
    btn_label = "🚀 " + " → ".join(parts) if parts else "✅ Nada pendiente"

    if st.button(btn_label, type="primary", key="btn_run_all", disabled=not parts):
        log_area = st.empty()
        all_ok = True

        for model in hpo_to_run:
            st.info(f"HPO: **{model}** — {all_n_trials} trials × {all_hpo_epochs} epochs")
            cmd_hpo = [
                PYTHON, "-m", "src.experiments.hpo",
                "--model",      model,
                "--n_trials",   str(all_n_trials),
                "--hpo_epochs", str(all_hpo_epochs),
                "--data_root",  data_root,
                "--seed",       str(seed),
                "--db",         hpo_db_path,
            ]
            rc, _, hpo_info = _stream_proc(cmd_hpo, log_area, parse_hpo=True)
            if rc == 0 and hpo_info.get("best_params"):
                st.session_state.hpo_results[model] = hpo_info
                st.success(f"HPO {model} listo — Best F1: {hpo_info['best']:.4f}")
            else:
                st.warning(f"HPO {model} termino sin params optimos (exit {rc}). Usara defaults.")

        for model in models_to_train:
            params = _get_params_for(model, fallback_lr, fallback_wd, fallback_bs, fallback_ls)
            st.info(f"Entrenando **{model}** — {params.get('_source', 'default')}")
            cmd_train = [
                PYTHON, "-m", "src.experiments.runner",
                "--models",          model,
                "--epochs",          str(all_epochs),
                "--seed",            str(seed),
                "--data_root",       data_root,
                "--lr",              str(params.get("lr",              fallback_lr)),
                "--weight_decay",    str(params.get("weight_decay",    fallback_wd)),
                "--batch_size",      str(params.get("batch_size",      fallback_bs)),
                "--patience",        "8",
                "--label_smoothing", str(params.get("label_smoothing", fallback_ls)),
            ]
            rc, _, _ = _stream_proc(cmd_train, log_area)
            if rc != 0:
                st.error(f"❌ Fallo entrenando {model} (exit {rc}).")
                all_ok = False

        if all_ok:
            st.success("✅ Pipeline completo. Ve al tab Results.")
            st.balloons()
