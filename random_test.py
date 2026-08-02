"""
random_test.py — esegue N episodi con policy casuale per osservare l'ambiente e
raccogliere statistiche (stesso protocollo di output di evaluate.py, riusa gli
helper di utils.py).

Uso:
    python random_test.py                        # 10 episodi, primo verbose
    python random_test.py --verbose | --quiet
    python random_test.py --n-episodes 20 --seed 42
    python random_test.py --no-color > log.txt   # senza ANSI (per file)
    python random_test.py --save [--save-dir runs/random_test]

Con --save produce <save-dir>/random_test_seed<S>_steps.jsonl e ..._summaries.jsonl.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from envs.incident_response_env import IncidentResponseEnv
from utils import (
    action_to_str,
    supports_color,
    state_legend,
    format_state,
    format_step_log,
    summarize_episode,
    format_episode_summary,
    aggregate_episode_summaries,
    format_aggregate_summary,
)


# --- Parsing argomenti CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Esegue rollout casuali su IncidentResponseEnv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-episodes", type=int, default=10,
                   help="numero di episodi da eseguire (default: 10)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed riproducibile per env e action sampling "
                        "(default: 0)")

    # I tre flag di verbosita' sono mutuamente esclusivi; default = first-verbose
    verb = p.add_mutually_exclusive_group()
    verb.add_argument("--verbose", action="store_true",
                      help="mostra step-by-step di TUTTI gli episodi")
    verb.add_argument("--quiet", action="store_true",
                      help="mostra solo la sintesi aggregata finale")

    p.add_argument("--no-color", action="store_true",
                   help="disabilita codici ANSI nell'output")

    p.add_argument("--save", action="store_true",
                   help="salva log step-by-step e sintesi su file JSONL")
    p.add_argument("--save-dir", type=str, default="runs",
                   help="directory di output per --save (default: 'runs')")
    return p


# --- Esecuzione di un singolo episodio ---

def run_episode(env: IncidentResponseEnv, action_rng: np.random.Generator,
                episode_seed: int):
    """
    Esegue un episodio con policy casuale (azioni campionate da action_rng,
    separato dall'RNG interno dell'env per riproducibilita'). Restituisce
    (history, total_reward, terminated, truncated).
    """
    obs, info = env.reset(seed=episode_seed)

    history = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(action_rng.integers(0, env.action_space.n))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Snapshot POST-step
        # Due versioni: plain per il JSONL, color per il log a terminale.
        state_snapshot_plain = format_state(env, use_color=False)
        state_snapshot_color = format_state(env, use_color=True)

        history.append({
            "action":     action,
            "action_str": action_to_str(action),
            "reward":     float(reward),
            "info":       dict(info),  # copia (x robustezza)
            "state_snapshot_plain": state_snapshot_plain,
            "state_snapshot_color": state_snapshot_color,
        })

    return history, total_reward, terminated, truncated


# --- Salvataggio JSONL ---

def save_jsonl(history_per_episode: list, summaries: list,
               save_dir: Path, seed: int):
    """
    Salva <save_dir>/random_test_seed<S>_steps.jsonl (una riga per step) e
    ..._summaries.jsonl (una riga per episodio)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    steps_path     = save_dir / f"random_test_seed{seed}_steps.jsonl"
    summaries_path = save_dir / f"random_test_seed{seed}_summaries.jsonl"

    with steps_path.open("w") as f:
        for ep_id, history in enumerate(history_per_episode):
            for step_n, record in enumerate(history):
                row = {
                    "episode": ep_id,
                    "step": step_n,
                    "action": record["action"],
                    "action_str": record["action_str"],
                    "reward": record["reward"],
                    "info": record["info"],
                    "state_snapshot": record["state_snapshot_plain"],  # senza ANSI
                }
                f.write(json.dumps(row) + "\n")

    with summaries_path.open("w") as f:
        for ep_id, summary in enumerate(summaries):
            row = {"episode": ep_id, **summary}
            f.write(json.dumps(row) + "\n")

    return steps_path, summaries_path


# --- Main ---

def main():
    args = build_parser().parse_args()

    use_color = supports_color(force_off=args.no_color)

    verbose       = args.verbose
    quiet         = args.quiet
    first_verbose = not (verbose or quiet)

    # RNG delle azioni ed env seedati dal seed globale: esecuzione riproducibile,
    # con patient_zero e dinamica diversi per episodio (episode_seed da master_rng)
    master_rng = np.random.default_rng(args.seed)
    env = IncidentResponseEnv(seed=args.seed)

    # Header: configurazione e legenda
    if not quiet:
        print(f"random_test — seed={args.seed}, n_episodes={args.n_episodes}, "
              f"colori={'on' if use_color else 'off'}")
        print()
        print(state_legend(use_color=use_color))
        print()

    history_per_episode = []
    summaries           = []

    for ep_id in range(args.n_episodes):
        episode_seed = int(master_rng.integers(0, 2**31 - 1))
        log_steps = verbose or (first_verbose and ep_id == 0)

        if log_steps:
            print(f"--- Episodio #{ep_id} (seed={episode_seed}) ---")

        history, total_reward, terminated, truncated = run_episode(
            env, master_rng, episode_seed
        )

        if log_steps:
            for record in history:
                snapshot = (record["state_snapshot_color"] if use_color
                            else record["state_snapshot_plain"])
                line = format_step_log(
                    step_n=record["info"]["current_step"],
                    action=record["action"],
                    reward=record["reward"],
                    info=record["info"],
                    state_snapshot=snapshot,
                    use_color=use_color,
                )
                print(line)
            print()

        # summarize_episode lavora sulla sola lista di dict info
        info_history = [r["info"] for r in history]
        summary = summarize_episode(info_history, total_reward,
                                    terminated, truncated, env)

        history_per_episode.append(history)
        summaries.append(summary)

        if not quiet:
            print(format_episode_summary(summary, episode_id=ep_id,
                                         use_color=use_color))
            print()

    # Sintesi aggregata finale
    agg = aggregate_episode_summaries(summaries)
    print(format_aggregate_summary(agg, use_color=use_color))

    if args.save:
        save_dir = Path(args.save_dir)
        steps_path, summaries_path = save_jsonl(
            history_per_episode, summaries, save_dir, args.seed
        )
        print()
        print(f"Salvato: {steps_path}")
        print(f"Salvato: {summaries_path}")


if __name__ == "__main__":
    main()
