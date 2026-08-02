"""
tools/plot_training_curves.py — grafici delle curve di training e tabella
numerica riassuntiva, a partire dagli evaluations.npz salvati da SB3.

Per ogni algoritmo (--algo dqn ppo) scandisce
<runs-root>/<algo>/full/seed*/eval_logs/evaluations.npz e produce: un PNG con le
5 curve per-seed + media in grassetto e banda +/-1 std; se sono presenti sia dqn
sia ppo, un PNG comparativo con le sole medie; una tabella .txt con peak reward,
step del peak, reward finale e n. evaluation per ciascun seed (per far quadrare i
grafici con runs_summary.json e citare valori puntuali nei diari).

Denormalizzazione: DQN è addestrato con reward/17, PPO no, quindi le curve
stanno su scale diverse. Se meta.json indica normalize_reward=true, moltiplico
le reward per il denominatore per riportarle alla scala naturale e rendere
le curve confrontabili.

Uso:
    python tools/plot_training_curves.py --algo dqn ppo \\
        --runs-root runs --out-dir runs/final_comparison

Output (sotto --out-dir, default runs/final_comparison): training_curves_dqn.png,
training_curves_ppo.png, training_curves_dqn_vs_ppo.png (se entrambi),
training_curves_summary.txt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")              # backend non-interattivo per server/script
import matplotlib.pyplot as plt


# Lettura di una singola run di training

def load_run(run_dir: Path) -> dict | None:
    """
    Legge una run (es. runs/dqn/full/seed1/) e restituisce un dict con timesteps,
    means (reward media per eval), normalize, denom, raw_results. Restituisce None
    se evaluations.npz o meta.json mancano.
    """
    npz_path  = run_dir / "eval_logs" / "evaluations.npz"
    meta_path = run_dir / "meta.json"

    if not npz_path.exists() or not meta_path.exists():
        return None

    data = np.load(npz_path)
    timesteps   = data["timesteps"]                # (n_evals,)
    raw_results = data["results"]                  # (n_evals, n_eval_eps)
    means       = raw_results.mean(axis=1)         # (n_evals,)

    with meta_path.open() as f:
        meta = json.load(f)
    normalize = bool(meta.get("normalize_reward", False))
    denom     = meta.get("reward_scale_denominator", None)

    # Denormalizzazione: riporta le reward alla scala naturale, per poter mettere
    # algoritmi diversi nello stesso grafico
    if normalize and denom is not None:
        means       = means * float(denom)
        raw_results = raw_results * float(denom)

    return {
        "timesteps":   timesteps,
        "means":       means,
        "normalize":   normalize,
        "denom":       float(denom) if denom is not None else None,
        "raw_results": raw_results,
    }


def load_algo_runs(algo: str, runs_root: Path) -> list[tuple[int, dict]]:
    """
    Carica tutte le run di un algoritmo da <runs-root>/<algo>/full/seed*/ e le
    restituisce come lista (seed, run_dict) ordinata per seed. Le run incomplete
    (load_run -> None) e le cartelle non "seedN" sono saltate.
    """
    full_dir = runs_root / algo / "full"
    if not full_dir.exists():
        return []

    seed_dirs = sorted(full_dir.glob("seed*"))
    runs = []
    for sd in seed_dirs:
        seed_str = sd.name.replace("seed", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue   # non e' una cartella "seedN"
        info = load_run(sd)
        if info is not None:
            runs.append((seed, info))
    return runs


# Plot per singolo algoritmo

def plot_algo_curves(algo: str, runs: list[tuple[int, dict]],
                     out_path: Path, title: str = None) -> None:
    """
    PNG con una curva per seed, la media in grassetto e una banda +/-1
    std. Le curve dei seed vengono interpolate su una griglia comune di timestep
    per gestire eventuali evaluation registrate in punti diversi
    (per le run ufficiali l'eval_freq e' identica, quindi coincidono gia')
    """
    if not runs:
        print(f"[plot_algo_curves] {algo}: nessuna run caricata, skip.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    all_t = np.unique(np.concatenate([info["timesteps"] for _, info in runs]))
    all_t = np.sort(all_t)

    aligned = []
    for seed, info in runs:
        ax.plot(info["timesteps"], info["means"],
                alpha=0.35, linewidth=1.0,
                label=f"seed {seed}")
        aligned.append(np.interp(all_t, info["timesteps"], info["means"]))

    arr = np.stack(aligned, axis=0)             # (n_seeds, n_grid)
    mean_curve = arr.mean(axis=0)
    std_curve  = arr.std(axis=0)

    ax.fill_between(all_t, mean_curve - std_curve, mean_curve + std_curve,
                    color="black", alpha=0.10, label="± 1 std")
    ax.plot(all_t, mean_curve, color="black", linewidth=2.2,
            label="media sui seed")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Reward media di evaluation (scala naturale)")
    ax.set_title(title or f"Curve di training — {algo.upper()}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Salvato: {out_path}")


# Plot di comparazione DQN vs PPO

def plot_algos_comparison(algos_data: dict, out_path: Path) -> None:
    """
    PNG con le sole curve medie dei due algoritmi (piu' banda di varianza fra
    seed), per confrontare direttamente DQN e PPO.
    """
    if not algos_data or all(len(v) == 0 for v in algos_data.values()):
        print("[plot_algos_comparison] nessun dato, skip.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    color_map = {"dqn": "#1f77b4", "ppo": "#d62728"}  

    for algo, runs in algos_data.items():
        if not runs:
            continue
        all_t = np.unique(np.concatenate([info["timesteps"] for _, info in runs]))
        all_t = np.sort(all_t)
        aligned = [np.interp(all_t, info["timesteps"], info["means"])
                   for _, info in runs]
        arr = np.stack(aligned, axis=0)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0)

        color = color_map.get(algo, None)
        ax.fill_between(all_t, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.15)
        ax.plot(all_t, mean_curve, color=color, linewidth=2.4,
                label=f"{algo.upper()} (media {len(runs)} seed)")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Reward media di evaluation (scala naturale)")
    ax.set_title("Curve di training — confronto fra algoritmi")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Salvato: {out_path}")


# Tabella numerica riassuntiva

def build_summary_table(algos_data: dict) -> str:
    """
    Tabella ASCII con i numeri chiave per ogni (algoritmo, seed): peak_reward,
    peak_step, final_reward, n_evals, normalize_reward. E' il complemento
    testuale dei grafici.
    """
    lines = []
    header = (f"{'algo':<6}{'seed':<6}{'peak_reward':<14}{'peak_step':<14}"
              f"{'final_reward':<16}{'n_evals':<10}{'normalized':<12}")
    lines.append(header)
    lines.append("-" * len(header))

    for algo, runs in algos_data.items():
        for seed, info in runs:
            ts    = info["timesteps"]
            means = info["means"]
            if ts.size == 0:
                continue
            idx_peak = int(np.argmax(means))
            row = (f"{algo:<6}"
                   f"{seed:<6}"
                   f"{means[idx_peak]:<14.3f}"
                   f"{int(ts[idx_peak]):<14d}"
                   f"{means[-1]:<16.3f}"
                   f"{len(means):<10d}"
                   f"{str(info['normalize']):<12}")
            lines.append(row)
        lines.append("")   # riga bianca fra algoritmi

    return "\n".join(lines)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plotta le curve di training di DQN e PPO dai .npz salvati."
    )
    p.add_argument("--algo", nargs="+", default=["dqn", "ppo"],
                   choices=["dqn", "ppo"],
                   help="quali algoritmi includere (default: dqn ppo)")
    p.add_argument("--runs-root", type=Path, default=Path("runs"),
                   help="directory radice delle run di training (default: runs)")
    p.add_argument("--out-dir", type=Path, default=Path("runs/final_comparison"),
                   help="directory di output (default: runs/final_comparison)")
    return p


def main():
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Carica tutti i dati prima di disegnare, cosi' la tabella vede gli stessi
    # seed dei grafici anche se qualche run fosse incompleta
    algos_data = {}
    for algo in args.algo:
        runs = load_algo_runs(algo, args.runs_root)
        algos_data[algo] = runs
        print(f"[{algo}] caricate {len(runs)} run da {args.runs_root}/{algo}/full")

    # Grafici per-algoritmo
    for algo, runs in algos_data.items():
        out_png = args.out_dir / f"training_curves_{algo}.png"
        plot_algo_curves(algo, runs, out_png)

    # Grafico comparativo (solo se ci sono almeno 2 algoritmi)
    if len([a for a, r in algos_data.items() if r]) >= 2:
        out_png = args.out_dir / "training_curves_dqn_vs_ppo.png"
        plot_algos_comparison(algos_data, out_png)

    # Tabella numerica
    summary = build_summary_table(algos_data)
    summary_path = args.out_dir / "training_curves_summary.txt"
    summary_path.write_text(summary)
    print(f"Salvato: {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
