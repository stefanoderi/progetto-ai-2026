"""
evaluate.py — orchestratore: esegue N episodi con una o piu' policy, raccoglie
metriche aggregate e produce log step-by-step opzionali.

Utilizzo (python evaluate.py --help):
    python evaluate.py --policy all --n-episodes 50
    python evaluate.py --policy dqn --model-path <best_model.zip> \\
        --n-episodes 50 --seed 42 --quiet
    python evaluate.py --policy heuristic,do_nothing,oracle,panic_isolation,dqn,ppo \\
        --model-path <dqn.zip> --ppo-model-path <ppo.zip> \\
        --n-episodes 50 --seed 42 --quiet --no-color

Output: per ogni policy la sintesi per-episodio + aggregato; con piu' policy,
una tabella di confronto finale. 

Con --save, per ogni policy due file JSONL in <save-dir>: 
    eval_<policy>_seed<S>_steps.jsonl e ..._summaries.jsonl.
"""
import argparse
import json
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

from baselines import (
    BasePolicy,
    RandomPolicy,
    DoNothingPolicy,
    PanicIsolationPolicy,
    HeuristicDefenderPolicy,
    OracleDefenderPolicy,
    SB3PolicyWrapper,
)


# --- Registry delle policy disponibili ---
# "dqn" e "ppo" hanno valore None: sono casi speciali che caricano un modello
# SB3 da disco (via --model-path / --ppo-model-path), gestiti in make_policy()
POLICY_REGISTRY = {
    "random":          RandomPolicy,
    "do_nothing":      DoNothingPolicy,
    "panic_isolation": PanicIsolationPolicy,
    "heuristic":       HeuristicDefenderPolicy,
    "oracle":          OracleDefenderPolicy,
    "dqn":             None,   # caricato da --model-path
    "ppo":             None,   # caricato da --ppo-model-path
}

# Policy incluse in --policy all (escluse dqn/ppo, che richiedono un model_path)
ALL_POLICY_NAMES = [n for n, cls in POLICY_REGISTRY.items() if cls is not None]


def make_policy(name: str, seed: int = 0,
                dqn_model_path: str = None,
                ppo_model_path: str = None) -> BasePolicy:
    """
    Factory: istanzia una policy dal nome registrato. seed serve solo a
    RandomPolicy; dqn_model_path/ppo_model_path solo a "dqn"/"ppo" (path del
    .zip SB3). 
    """
    if name == "dqn":
        if dqn_model_path is None:
            raise ValueError(
                "policy 'dqn' richiede --model-path verso il file .zip "
                "del modello Stable-Baselines3."
            )
        # SB3 la importiamo solo quando serve per non pagarne il costo sulle sole baseline
        from stable_baselines3 import DQN
        model = DQN.load(dqn_model_path)
        # nome leggibile dqn[<file>] per distinguere le run in tabella
        return SB3PolicyWrapper(
            model,
            deterministic=True,
            name=f"dqn[{Path(dqn_model_path).name}]",
        )

    if name == "ppo":
        if ppo_model_path is None:
            raise ValueError(
                "policy 'ppo' richiede --ppo-model-path verso il file .zip "
                "del modello Stable-Baselines3."
            )
        from stable_baselines3 import PPO
        model = PPO.load(ppo_model_path)
        return SB3PolicyWrapper(
            model,
            deterministic=True,
            name=f"ppo[{Path(ppo_model_path).name}]",
        )

    cls = POLICY_REGISTRY[name]
    if name == "random":
        return cls(seed=seed)
    return cls()


# --- Esecuzione di un singolo episodio ---

def run_episode(env: IncidentResponseEnv, policy: BasePolicy,
                episode_seed: int):
    """Esegue un episodio completo e restituisce (history, total_reward, terminated, truncated)."""
    obs, info = env.reset(seed=episode_seed)
    policy.reset()

    history = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Snapshot POST-step
        state_snapshot_plain = format_state(env, use_color=False)
        state_snapshot_color = format_state(env, use_color=True)

        history.append({
            "action":     action,
            "action_str": action_to_str(action),
            "reward":     float(reward),
            "info":       dict(info),  # copia difensiva
            "state_snapshot_plain": state_snapshot_plain,
            "state_snapshot_color": state_snapshot_color,
        })

    return history, total_reward, terminated, truncated


# --- Esecuzione di N episodi per una policy ---

def evaluate_policy(policy_name: str, args, use_color: bool,
                    dqn_model_path: str = None,
                    ppo_model_path: str = None):
    """
    Esegue args.n_episodes episodi con la policy indicata e restituisce
    (history_per_episode, summaries).

    A parita' di --seed ogni policy vede la stessa sequenza di scenari
    (patient_zero e propagazione), confronto fra policy fair
    """
    env    = IncidentResponseEnv(seed=args.seed)
    policy = make_policy(policy_name, seed=args.seed,
                         dqn_model_path=dqn_model_path,
                         ppo_model_path=ppo_model_path)
    master_rng = np.random.default_rng(args.seed)

    verbose       = args.verbose
    quiet         = args.quiet
    first_verbose = not (verbose or quiet)

    history_per_episode = []
    summaries           = []

    for ep_id in range(args.n_episodes):
        episode_seed = int(master_rng.integers(0, 2**31 - 1))
        log_steps    = verbose or (first_verbose and ep_id == 0)

        if log_steps:
            print(f"--- Episodio #{ep_id} (seed={episode_seed}) ---")

        history, total_reward, terminated, truncated = run_episode(
            env, policy, episode_seed
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

    return history_per_episode, summaries


# --- Salvataggio JSONL ---

def save_jsonl(history_per_episode, summaries, save_dir: Path,
               seed: int, policy_name: str):
    """
    Salva due file JSONL in save_dir: eval_<policy>_seed<S>_steps.jsonl (una
    riga per step) e ..._summaries.jsonl (una riga per episodio).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    # sanitizza [ ] / nel nome (es. dqn[best_model.zip]) per non rompere il path
    safe_name = policy_name.replace("[", "_").replace("]", "_").replace("/", "_")

    steps_path     = save_dir / f"eval_{safe_name}_seed{seed}_steps.jsonl"
    summaries_path = save_dir / f"eval_{safe_name}_seed{seed}_summaries.jsonl"

    with steps_path.open("w") as f:
        for ep_id, history in enumerate(history_per_episode):
            for step_n, record in enumerate(history):
                row = {
                    "episode":    ep_id,
                    "step":       step_n,
                    "action":     record["action"],
                    "action_str": record["action_str"],
                    "reward":     record["reward"],
                    "info":       record["info"],
                    "state_snapshot": record["state_snapshot_plain"],
                }
                f.write(json.dumps(row) + "\n")

    with summaries_path.open("w") as f:
        for ep_id, summary in enumerate(summaries):
            row = {"episode": ep_id, **summary}
            f.write(json.dumps(row) + "\n")

    return steps_path, summaries_path


# --- Tabella di confronto fra piu' policy ---

def extract_metrics(agg: dict, summaries: list) -> dict:
    """
    Metriche per la tabella di confronto. I tassi si leggono
    dall'aggregate; reward, step e valore compromesso medio si ricalcolano dai
    summaries con numpy. Le chiavi usate sono quelle prodotte da
    utils (aggregate_episode_summaries e summarize_episode).
    """
    metrics = {
        "terminated_rate": float(agg["terminated_rate"]),
        "truncated_rate":  float(agg["truncated_rate"]),
        "win_rate":        float(agg["win_rate"]),
        "wipeout_rate":    float(agg["wipeout_rate"]),
        "invalid_rate":    float(agg["mean_invalid_action_rate"]),
        "critical_rate":   float(agg["critical_compromised_rate"]),
    }

    rewards    = [float(s["total_reward"])           for s in summaries]
    steps      = [float(s["n_steps"])                for s in summaries]
    comp_value = [float(s["mean_compromised_value"]) for s in summaries]

    metrics["reward_mean"]            = float(np.mean(rewards))    if rewards    else 0.0
    metrics["reward_std"]             = float(np.std(rewards))     if rewards    else 0.0
    metrics["steps_mean"]             = float(np.mean(steps))      if steps      else 0.0
    metrics["compromised_value_mean"] = float(np.mean(comp_value)) if comp_value else 0.0
    return metrics


def format_comparison_table(metrics_per_policy: dict) -> str:
    """
    Tabella ASCII di confronto fra piu' policy. Colonne:
        policy       nome della policy
        reward_mean  reward media (alta = meglio)
        reward_std   deviazione standard della reward (bassa = piu' stabile)
        steps_mean   step medi per episodio (da leggere con win/wipe: episodi
                     corti = vittoria veloce oppure wipeout veloce)
        term%        episodi terminati (vittoria o wipeout)
        win%         di cui vittorie (rete bonificata, clean_streak target)
        wipe%        di cui sconfitte totali (tutti i nodi compromessi)
        trunc%       episodi finiti per limite di step
        inv%         tasso medio di azioni invalide
        crit%        episodi con il critical compromesso almeno una volta
        comp_val     valore operativo medio compromesso per step (basso = meglio)
    """
    headers = ["policy", "reward_mean", "reward_std", "steps_mean",
               "term%", "win%", "wipe%", "trunc%", "inv%", "crit%",
               "comp_val"]

    rows = []
    for name, m in metrics_per_policy.items():
        rows.append([
            name,
            f"{m['reward_mean']:.2f}",
            f"{m['reward_std']:.2f}",
            f"{m['steps_mean']:.1f}",
            f"{100 * m['terminated_rate']:.1f}",
            f"{100 * m['win_rate']:.1f}",
            f"{100 * m['wipeout_rate']:.1f}",
            f"{100 * m['truncated_rate']:.1f}",
            f"{100 * m['invalid_rate']:.1f}",
            f"{100 * m['critical_rate']:.1f}",
            f"{m['compromised_value_mean']:.2f}",
        ])

    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def fmt(row):
        return "  ".join(str(x).ljust(w) for x, w in zip(row, widths))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


# --- Parsing della lista di policy ---

def parse_policy_arg(policy_arg: str) -> list:
    """
    Risolve --policy in una lista di nomi validi: "all" -> tutte le baseline
    (esclude dqn/ppo, che richiedono un model_path); altrimenti un nome singolo
    o una lista comma-separated. Solleva ValueError sui nomi non riconosciuti.
    """
    if policy_arg == "all":
        return list(ALL_POLICY_NAMES)

    names = [n.strip() for n in policy_arg.split(",")]
    unknown = [n for n in names if n not in POLICY_REGISTRY]
    if unknown:
        raise ValueError(
            f"Policy non riconosciute: {unknown}. "
            f"Valide: {list(POLICY_REGISTRY.keys()) + ['all']}"
        )
    return names


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Esegue evaluation con una o piu' policy difensive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --policy non usa choices=: accetta anche liste comma-separated, validate
    # a valle da parse_policy_arg
    p.add_argument("--policy", type=str, default="all",
                   help="policy da valutare. Valori: 'all' (tutte le baseline), "
                        "un nome singolo, o una lista comma-separated "
                        "(es. 'heuristic,dqn,ppo'). Default: 'all'.")
    p.add_argument("--n-episodes", type=int, default=10,
                   help="numero di episodi per policy (default: 10)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed riproducibile, condiviso fra policy "
                        "(default: 0)")

    p.add_argument("--model-path", type=str, default=None,
                   help="path al file .zip del modello DQN, obbligatorio "
                        "quando 'dqn' e' fra le policy selezionate.")
    p.add_argument("--ppo-model-path", type=str, default=None,
                   help="path al file .zip del modello PPO, obbligatorio "
                        "quando 'ppo' e' fra le policy selezionate.")

    verb = p.add_mutually_exclusive_group()
    verb.add_argument("--verbose", action="store_true",
                      help="mostra step-by-step di TUTTI gli episodi")
    verb.add_argument("--quiet", action="store_true",
                      help="mostra solo aggregati e tabella di confronto")

    p.add_argument("--no-color", action="store_true",
                   help="disabilita codici ANSI nell'output")

    p.add_argument("--save", action="store_true",
                   help="salva log step-by-step e sintesi su file JSONL")
    p.add_argument("--save-dir", type=str, default="runs",
                   help="directory di output per --save (default: 'runs')")
    return p


def main():
    args = build_parser().parse_args()
    use_color = supports_color(force_off=args.no_color)

    policy_names = parse_policy_arg(args.policy)

    # se 'dqn'/'ppo' e' fra le policy, serve il rispettivo model-path
    if "dqn" in policy_names and args.model_path is None:
        raise SystemExit(
            "Errore: --model-path e' obbligatorio quando 'dqn' e' fra le "
            "policy selezionate.\n"
            "Esempio: python evaluate.py --policy dqn "
            "--model-path runs/dqn/full/seed0/best_model/best_model.zip"
        )
    if "ppo" in policy_names and args.ppo_model_path is None:
        raise SystemExit(
            "Errore: --ppo-model-path e' obbligatorio quando 'ppo' e' fra le "
            "policy selezionate.\n"
            "Esempio: python evaluate.py --policy ppo "
            "--ppo-model-path runs/ppo/full/seed0/best_model/best_model.zip"
        )

    save_dir = Path(args.save_dir) if args.save else None
    metrics_per_policy = {}

    for name in policy_names:
        print()
        print(f"========== Policy: {name} ==========")
        if not args.quiet:
            print(f"seed={args.seed}, n_episodes={args.n_episodes}")
            if name == "dqn":
                print(f"model_path={args.model_path}")
            if name == "ppo":
                print(f"ppo_model_path={args.ppo_model_path}")
            print()
            print(state_legend(use_color=use_color))
            print()

        history_per_episode, summaries = evaluate_policy(
            name, args, use_color,
            dqn_model_path=args.model_path     if name == "dqn" else None,
            ppo_model_path=args.ppo_model_path if name == "ppo" else None,
        )

        agg = aggregate_episode_summaries(summaries)
        # nome esteso (dqn[file]/ppo[file]) per distinguere le run in tabella
        display_name = name
        if name == "dqn":
            display_name = f"dqn[{Path(args.model_path).name}]"
        elif name == "ppo":
            display_name = f"ppo[{Path(args.ppo_model_path).name}]"
        metrics_per_policy[display_name] = extract_metrics(agg, summaries)

        print(format_aggregate_summary(agg, use_color=use_color))

        if save_dir is not None:
            steps_path, summaries_path = save_jsonl(
                history_per_episode, summaries, save_dir,
                args.seed, display_name,
            )
            print()
            print(f"Salvato: {steps_path}")
            print(f"Salvato: {summaries_path}")

    # Tabella di confronto solo se sono state eseguite piu' policy
    if len(policy_names) > 1:
        print()
        print("========== Tabella di confronto ==========")
        print(format_comparison_table(metrics_per_policy))


if __name__ == "__main__":
    main()