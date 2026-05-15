from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.core.config import TrainingConfig
from src.core.registry import build_model, list_models
from src.training.classical_trainer import build_trainer
from src.utils.seed import set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SEARCH_SPACE: dict[str, tuple] = {
    "lr":               ("log_float", 1e-5, 1e-2),
    "weight_decay":     ("log_float", 1e-5, 1e-2),
    "batch_size":       ("categorical", [32, 64, 128]),
    "label_smoothing":  ("float", 0.0, 0.2),
    "backbone_lr_mult": ("float", 0.01, 0.5),
    "warmup_epochs":    ("int", 1, 5),
}

def _suggest_params(trial: optuna.Trial) -> dict:
    params = {}
    for name, spec in _SEARCH_SPACE.items():
        kind, *args = spec
        if kind == "log_float":
            params[name] = trial.suggest_float(name, *args, log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, *args)
        elif kind == "int":
            params[name] = trial.suggest_int(name, *args)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, args[0])
    return params

def make_objective(model_name: str, data_root: str, hpo_epochs: int, seed: int):

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)

        config = TrainingConfig(
            data_root=data_root,
            epochs=hpo_epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            label_smoothing=params["label_smoothing"],
            backbone_lr_mult=params["backbone_lr_mult"],
            warmup_epochs=params["warmup_epochs"],
            patience=hpo_epochs + 1,
            checkpoint_dir="results/hpo_tmp",
            results_dir="results/hpo_tmp",
        )

        set_seed(seed + trial.number)
        model = build_model(model_name, num_classes=config.num_classes, pretrained=True)
        trainer, _ = build_trainer(model, f"{model_name}_t{trial.number}", config)
        return trainer.fit_hpo(trial)

    return objective

def run_study(
    model_name: str,
    data_root: str,
    n_trials: int,
    hpo_epochs: int,
    seed: int,
    db_path: str = "results/optuna.db",
) -> optuna.Study:
    Path("results").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, n_startup_trials=max(10, n_trials // 5)),
        pruner=HyperbandPruner(
            min_resource=3,
            max_resource=hpo_epochs,
            reduction_factor=3,
        ),
        study_name=f"hpo_{model_name}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    def _log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best = study.best_trial
            print(
                f"[TRIAL_DONE] trial={trial.number} "
                f"value={trial.value:.4f} "
                f"best={best.value:.4f}",
                flush=True,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"[TRIAL_PRUNED] trial={trial.number}", flush=True)

    study.optimize(
        make_objective(model_name, data_root, hpo_epochs, seed),
        n_trials=n_trials,
        callbacks=[_log_callback],
        catch=(Exception,),
    )

    print(f"\n[HPO_DONE] best_value={study.best_value:.4f}", flush=True)
    print(f"[BEST_PARAMS] {json.dumps(study.best_params)}", flush=True)
    return study

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Optuna HPO for QuantumCafe-CR classical models")
    p.add_argument("--model", required=True, choices=list_models("classical"),
                   help="Classical model to optimize.")
    p.add_argument("--n_trials", type=int, default=30,
                   help="Total Optuna trials to run.")
    p.add_argument("--hpo_epochs", type=int, default=15,
                   help="Epochs per trial (more = more accurate signal, fewer trials feasible).")
    p.add_argument("--data_root", default="data/raw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", default="results/optuna.db",
                   help="SQLite path for Optuna study persistence (allows resuming).")
    args = p.parse_args()

    set_seed(args.seed)
    study = run_study(
        model_name=args.model,
        data_root=args.data_root,
        n_trials=args.n_trials,
        hpo_epochs=args.hpo_epochs,
        seed=args.seed,
        db_path=args.db,
    )

    print("\n=== HPO SUMMARY ===")
    print(f"Best val macro-F1 : {study.best_value:.4f}")
    print(f"Best params        : {json.dumps(study.best_params, indent=2)}")
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"Trials completed   : {completed} | pruned: {pruned}")
