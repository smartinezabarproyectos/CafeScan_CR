"""train_overnight.py — Corre todo el pipeline de entrenamiento sin intervencion.

Configura la seccion CONFIG de abajo y ejecuta:
    python scripts/train_overnight.py

El script corre en orden:
    1. HPO (opcional) para cada modelo configurado
    2. Entrenamiento completo de todos los modelos con mejores params
    3. Reporte de comparacion final

Logs guardados en: results/overnight_YYYY-MM-DD_HH-MM.log
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edita aqui antes de correr
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Datos ─────────────────────────────────────────────────────────────────
    "data_root":   "data/raw",
    "results_dir": "results",
    "seed":        42,

    # ── HPO ───────────────────────────────────────────────────────────────────
    # Modelos que reciben busqueda HPO independiente.
    # Usa [] para saltar HPO y usar los hiperparametros por defecto.
    "hpo_models":   ["efficientnet_b0", "vit", "mobilenet"],
    "hpo_n_trials": 50,
    "hpo_epochs":   20,
    "hpo_db":       "results/optuna.db",

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    # resnet50 hereda los mejores params de efficientnet_b0 si no tiene HPO propio.
    "models":  ["efficientnet_b0", "resnet50", "vit", "mobilenet"],
    "epochs":  40,

    # Defaults si no se corre HPO (o para modelos sin HPO propio)
    "default_lr":              1e-4,
    "default_weight_decay":    1e-4,
    "default_batch_size":      64,
    "default_patience":        8,
    "default_label_smoothing": 0.1,

    # ── Misc ──────────────────────────────────────────────────────────────────
    "skip_hpo": False,
}

# ═══════════════════════════════════════════════════════════════════════════════
# No edites debajo de esta linea
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


def setup_logging() -> logging.Logger:
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = Path(CONFIG["results_dir"]) / f"overnight_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    log = logging.getLogger("overnight")
    log.info(f"Log guardado en: {log_file}")
    return log


def run_cmd(cmd: list[str], log: logging.Logger) -> int:
    env = {"PYTHONUNBUFFERED": "1", **{k: v for k, v in __import__("os").environ.items()}}
    log.info(f"CMD: {' '.join(str(c) for c in cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info(line)
    proc.wait()
    return proc.returncode


def load_hpo_from_db(model: str, log: logging.Logger) -> dict | None:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        db_path = Path(PROJECT_ROOT) / CONFIG["hpo_db"]
        if not db_path.exists():
            return None
        study = optuna.load_study(
            study_name=f"hpo_{model}",
            storage=f"sqlite:///{db_path}",
        )
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            return None
        log.info(f"[DB] {model}: {len(completed)} trials — best F1={study.best_value:.4f}")
        log.info(f"[DB] {model} best params: {study.best_params}")
        return study.best_params
    except Exception as e:
        log.warning(f"No se pudo cargar HPO de '{model}' desde DB: {e}")
        return None


def run_hpo(model: str, log: logging.Logger) -> dict | None:
    log.info(f"\n{'='*60}")
    log.info(f"HPO: {model} | {CONFIG['hpo_n_trials']} trials x {CONFIG['hpo_epochs']} epochs")
    log.info(f"{'='*60}")

    cmd = [
        PYTHON, "-m", "src.experiments.hpo",
        "--model",      model,
        "--n_trials",   str(CONFIG["hpo_n_trials"]),
        "--hpo_epochs", str(CONFIG["hpo_epochs"]),
        "--data_root",  CONFIG["data_root"],
        "--seed",       str(CONFIG["seed"]),
        "--db",         CONFIG["hpo_db"],
    ]

    rc = run_cmd(cmd, log)
    if rc != 0:
        log.error(f"HPO de {model} fallo (exit {rc}). Usando params por defecto.")
        return None

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        db_path = Path(PROJECT_ROOT) / CONFIG["hpo_db"]
        study = optuna.load_study(
            study_name=f"hpo_{model}",
            storage=f"sqlite:///{db_path}",
        )
        log.info(f"Mejor F1 HPO ({model}): {study.best_value:.4f}")
        log.info(f"Mejores params: {study.best_params}")
        return study.best_params
    except Exception as e:
        log.error(f"No se pudo leer el estudio de Optuna: {e}")
        return None


def run_training(models: list[str], params: dict, epochs: int, log: logging.Logger) -> bool:
    log.info(f"\n{'='*60}")
    log.info(f"TRAINING: {models} | {epochs} epochs")
    log.info(f"Params: {params}")
    log.info(f"{'='*60}")

    cmd = [
        PYTHON, "-m", "src.experiments.runner",
        "--models", *models,
        "--epochs",          str(epochs),
        "--seed",            str(CONFIG["seed"]),
        "--data_root",       CONFIG["data_root"],
        "--lr",              str(params.get("lr",              CONFIG["default_lr"])),
        "--weight_decay",    str(params.get("weight_decay",    CONFIG["default_weight_decay"])),
        "--batch_size",      str(params.get("batch_size",      CONFIG["default_batch_size"])),
        "--patience",        str(params.get("patience",        CONFIG["default_patience"])),
        "--label_smoothing", str(params.get("label_smoothing", CONFIG["default_label_smoothing"])),
    ]

    rc = run_cmd(cmd, log)
    if rc != 0:
        log.error(f"Training fallo (exit {rc})")
        return False
    return True


def _resolve_params(model: str, best_params: dict[str, dict]) -> dict:
    if model in best_params and best_params[model]:
        return best_params[model]
    fallback = best_params.get("efficientnet_b0", {})
    if fallback:
        return fallback
    return {}


def main():
    log = setup_logging()
    start = time.time()

    log.info("=" * 60)
    log.info("QUANTUMCAFE-CR — OVERNIGHT TRAINING")
    log.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)
    log.info(f"Config:\n{json.dumps(CONFIG, indent=2)}")

    best_params: dict[str, dict] = {}

    # ── Fase 0: Cargar HPO ya existentes del DB ───────────────────────────────
    log.info("\n\n>>> FASE 0: CARGANDO HPO EXISTENTES DEL DB")
    for model in CONFIG["models"]:
        if model not in CONFIG["hpo_models"]:
            params = load_hpo_from_db(model, log)
            if params:
                best_params[model] = params
            else:
                log.info(f"  Sin HPO previo para {model} — usara defaults o hereda de efficientnet_b0")

    # ── Fase 1: HPO para modelos pendientes ───────────────────────────────────
    if not CONFIG["skip_hpo"] and CONFIG["hpo_models"]:
        log.info("\n\n>>> FASE 1: HPO")
        for model in CONFIG["hpo_models"]:
            params = run_hpo(model, log)
            best_params[model] = params if params else {}
    else:
        log.info("\n\n>>> FASE 1: HPO omitido — usando hiperparametros por defecto")

    # ── Fase 2: Entrenamiento (uno por uno con sus propios params) ────────────
    log.info("\n\n>>> FASE 2: ENTRENAMIENTO")
    all_ok = True
    for model in CONFIG["models"]:
        params = _resolve_params(model, best_params)
        log.info(f"\n  → {model}: params = {params or 'defaults'}")
        ok = run_training([model], params, epochs=CONFIG["epochs"], log=log)
        if not ok:
            log.error(f"  Fallo entrenando {model}. Continuando con el siguiente.")
            all_ok = False

    if all_ok:
        log.info("Todos los modelos entrenados correctamente.")

    # ── Fase 3: Reporte final ─────────────────────────────────────────────────
    log.info("\n\n>>> FASE 3: REPORTE FINAL")
    cmd = [PYTHON, "-c", f"""
import sys; sys.path.insert(0, '.')
from src.evaluation.comparator import load_all_results, print_summary_table, plot_comparison, plot_per_class_f1
from pathlib import Path
df = load_all_results('{CONFIG["results_dir"]}')
if not df.empty:
    print_summary_table(df)
    fig = Path('{CONFIG["results_dir"]}') / 'figures'
    fig.mkdir(exist_ok=True)
    plot_comparison(df, fig / 'model_comparison.png')
    plot_per_class_f1(df, fig / 'per_class_f1.png')
    print('Figuras guardadas en results/figures/')
else:
    print('No se encontraron resultados en results/')
"""]
    run_cmd(cmd, log)

    elapsed = time.time() - start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    log.info(f"\n{'='*60}")
    log.info(f"OVERNIGHT TRAINING COMPLETO")
    log.info(f"Tiempo total: {h}h {m}m {s}s")
    log.info(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
