"""
train_dqn.py — addestramento di DQN su IncidentResponseEnv.

Tre modalita' via --mode, con budget e sintonia crescenti:
    smoke        ~10k step: verifica end-to-end che la pipeline giri e la reward
                 si muova (learning_starts abbassato a 1000).
    preliminary  ~200k step: osserva andamento reward, calo delle azioni
                 invalide e stabilita' prima della run full.
    full         500k step per seed, iperparametri = default SB3; con
                 --seeds 0,1,2,3,4 produce 5 run nella stessa cartella.

Output per run: <out_dir>/<mode>/seed<S>/ con best_model/best_model.zip,
eval_logs/evaluations.npz, final_model.zip, meta.json, tb_logs/.

Con piu' seed viene prodotto anche runs_summary.json (aggregati best/final mean +/- std).

Seed: env_seed (patient_zero, propagazione, rumore) e algo_seed (init rete,
replay sampling, epsilon-greedy) sono separati e registrati nei meta.json, per
distinguere la varianza da scenario da quella da algoritmo. Default:
algo_seed = env_seed + 1000.

Reward normalization (--normalize-reward): divide la reward per 17 (somma dei
business value). Non cambia la policy ottimale, porta la reward circa in [-1, 0] 
e migliora il condizionamento. Si usa solo se si osserva divergenza con la scala originale.

Esempi:
    python train_dqn.py --mode smoke --seed 42
    python train_dqn.py --mode full --seeds 0,1,2,3,4
    python train_dqn.py --mode full --seeds 0,1,2,3,4 --normalize-reward
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.incident_response_env import IncidentResponseEnv


# Normalizzazione reward (opzionale)
# 17 = somma di tutti i business value (5*1 + 2*3 + 1*6), da attivare solo se DQN diverge.
REWARD_SCALE_DENOMINATOR = 17.0


class RewardScaleWrapper(gym.RewardWrapper):
    """
    Divide la reward di ogni step per una costante fissa (default 17.0). Va
    applicato sia al train env sia all'eval env, altrimenti le curve di eval
    userebbero una scala diversa dal training e diventerebbero illeggibili
    """
    def __init__(self, env, denominator: float = REWARD_SCALE_DENOMINATOR):
        super().__init__(env)
        self.denominator = float(denominator)

    def reward(self, reward):
        return reward / self.denominator


# Iperparametri DQN (default Stable-Baselines3 v2.x)
@dataclass
class DQNHparams:
    """
    Iperparametri di DQN: tutti i default di SB3 v2.x tranne learning_starts.
    Il default v2.x (100) e' troppo poco qui (~3 episodi da max 40 step); usiamo
    5_000 (~125 episodi). In smoke anche 5_000 e' troppo alto:
    MODE_DEFAULTS["smoke"] lo abbassa a 1_000.
    """
    # Apprendimento
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000        # override solo in smoke
    batch_size: int = 32
    tau: float = 1.0                     # hard target update
    gamma: float = 0.99

    # Frequenza training
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000

    # Esplorazione epsilon-greedy
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # Regolarizzazione
    max_grad_norm: float = 10.0


# Configurazione delle modalita'

# Per ogni modalita': total_timesteps (budget), eval_freq (ogni quanti step
# l'eval periodica), n_eval_episodes, learning_starts_override (None = default)
MODE_DEFAULTS = {
    "smoke": {
        "total_timesteps":          10_000,
        "eval_freq":                 2_000,
        "n_eval_episodes":              10,
        "learning_starts_override":  1_000,   # serve a vedere la loss che scende
    },
    "preliminary": {
        "total_timesteps":         200_000,
        "eval_freq":                10_000,
        "n_eval_episodes":              20,
        "learning_starts_override":   None,   # usa il default della dataclass (5_000)
    },
    "full": {
        "total_timesteps":         500_000,
        "eval_freq":                25_000,
        "n_eval_episodes":              50,
        "learning_starts_override":   None,   # usa il default della dataclass (5_000)
    },
}


# Costruzione dell'ambiente
def make_env(seed: int, normalize_reward: bool):
    """
    Costruisce l'env Gymnasium pronto per SB3. Monitor e' il wrapper piu'
    esterno e va applicato sempre (train ed eval) per silenziare il warning di
    SB3 sull'eval env; registra la reward effettivamente vista dall'agente (già
    scalata quando RewardScaleWrapper e' attivo).
    """
    env = IncidentResponseEnv(seed=seed)
    if normalize_reward:
        env = RewardScaleWrapper(env)
    env = Monitor(env)
    return env


def _get_sb3_version() -> str:
    """Versione di stable-baselines3 (registrata in meta.json)"""
    try:
        import stable_baselines3 as sb3
        return sb3.__version__
    except Exception:
        return "unknown"


# Esecuzione di un singolo training con un singolo seed
def run_single_seed(
    mode: str,
    env_seed: int,
    algo_seed: int,
    output_dir: Path,
    normalize_reward: bool,
    verbose: int = 1,
) -> dict:
    """
    Esegue una run di training completa per (mode, env_seed, algo_seed) e
    restituisce un dict con env_seed, algo_seed, training_time_seconds,
    best_mean_reward_eval, final_mean_reward_eval, run_dir
    """
    cfg = MODE_DEFAULTS[mode]

    # Iperparametri: default SB3 + eventuale override di learning_starts
    hparams = DQNHparams()
    if cfg["learning_starts_override"] is not None:
        hparams.learning_starts = cfg["learning_starts_override"]

    # Directory di output
    run_dir  = output_dir / mode / f"seed{env_seed}"
    tb_dir   = run_dir / "tb_logs"
    eval_dir = run_dir / "eval_logs"
    best_dir = run_dir / "best_model"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Eval env con seed +10000 rispetto al train env: gli scenari
    # dell'eval periodica sono diversi da quelli del rollout di training,
    # cosi' si evita data leakage nella valutazione interna.
    env_train = make_env(seed=env_seed,
                         normalize_reward=normalize_reward)
    env_eval  = make_env(seed=env_seed + 10_000,
                         normalize_reward=normalize_reward)

    # Callback di eval periodica: salva il best model (miglior reward medio) in
    # best_dir/best_model.zip e la storia dei reward in eval_dir/evaluations.npz
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=0,
    )

    # ---- Modello DQN ----
    model = DQN(
        "MlpPolicy",
        env_train,
        learning_rate=hparams.learning_rate,
        buffer_size=hparams.buffer_size,
        learning_starts=hparams.learning_starts,
        batch_size=hparams.batch_size,
        tau=hparams.tau,
        gamma=hparams.gamma,
        train_freq=hparams.train_freq,
        gradient_steps=hparams.gradient_steps,
        target_update_interval=hparams.target_update_interval,
        exploration_fraction=hparams.exploration_fraction,
        exploration_initial_eps=hparams.exploration_initial_eps,
        exploration_final_eps=hparams.exploration_final_eps,
        max_grad_norm=hparams.max_grad_norm,
        seed=algo_seed,
        tensorboard_log=str(tb_dir),
        verbose=verbose,
    )

    # Training
    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=eval_callback,
        progress_bar=False,   # off
        log_interval=10,
    )
    t_elapsed = time.time() - t0

    # Salvataggio modello finale
    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))

    # ---- Metriche da evaluations.npz ----
    # results: (n_evals, n_eval_episodes) = reward dei singoli episodi di ogni
    # eval periodica; means = media per eval -> traiettoria della performance
    eval_npz_path = eval_dir / "evaluations.npz"
    best_mean = float("nan")
    final_mean = float("nan")
    if eval_npz_path.exists():
        data = np.load(eval_npz_path)
        results = data["results"]
        if results.size > 0:
            means = results.mean(axis=1)
            best_mean  = float(np.max(means))
            final_mean = float(means[-1])

    # meta.json: tutto cio' che serve a riprodurre la run
    meta = {
        "mode":                       mode,
        "algo_seed":                  algo_seed,
        "normalize_reward":           normalize_reward,
        "reward_scale_denominator":   REWARD_SCALE_DENOMINATOR if normalize_reward else None,
        "total_timesteps":            cfg["total_timesteps"],
        "eval_freq":                  cfg["eval_freq"],
        "n_eval_episodes":            cfg["n_eval_episodes"],
        "hparams":                    asdict(hparams),
        "training_time_seconds":      t_elapsed,
        "best_mean_reward_eval":      best_mean,
        "final_mean_reward_eval":     final_mean,
        "sb3_version":                _get_sb3_version(),
    }
    with (run_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "env_seed":                 env_seed,
        "algo_seed":                algo_seed,
        "training_time_seconds":    t_elapsed,
        "best_mean_reward_eval":    best_mean,
        "final_mean_reward_eval":   final_mean,
        "run_dir":                  str(run_dir),
    }


# --- CLI ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Addestra DQN su IncidentResponseEnv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", type=str, default="smoke",
                   choices=["smoke", "preliminary", "full"],
                   help="modalita' di training (default: smoke)")

    grp_seed = p.add_mutually_exclusive_group()
    grp_seed.add_argument("--seed", type=int, default=None,
                          help="seed singolo: env_seed=SEED, "
                               "algo_seed=SEED + algo-seed-offset")
    grp_seed.add_argument("--seeds", type=str, default=None,
                          help="seeds multipli separati da virgola, es. '0,1,2,3,4'. "
                               "Mutuamente esclusivo con --seed.")

    p.add_argument("--algo-seed-offset", type=int, default=1000,
                   help="offset per derivare algo_seed da env_seed "
                        "(algo_seed = env_seed + offset). Default: 1000.")

    p.add_argument("--output-dir", type=str, default="runs/dqn",
                   help="directory radice di output (default: 'runs/dqn')")

    p.add_argument("--normalize-reward", action="store_true",
                   help="divide la reward per 17 (somma BV) per migliorare "
                        "la stabilita' numerica di DQN. Usare solo se "
                        "si osserva divergenza con la scala originale.")

    p.add_argument("--verbose", type=int, default=1,
                   help="livello di verbosita' SB3 (0, 1, 2). Default: 1.")

    return p


def main():
    args = build_parser().parse_args()

    # Risoluzione lista seed
    if args.seeds is not None:
        env_seeds = [int(s.strip()) for s in args.seeds.split(",")]
    elif args.seed is not None:
        env_seeds = [args.seed]
    else:
        # default: 42 per smoke/preliminary, i 5 seed ufficiali per full
        env_seeds = [42] if args.mode in ("smoke", "preliminary") else [0, 1, 2, 3, 4]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== DQN training — mode={args.mode}, seeds={env_seeds} ===")
    print(f"output:           {output_dir / args.mode}")
    print(f"normalize_reward: {args.normalize_reward}")
    print(f"sb3 version:      {_get_sb3_version()}")
    print()

    # Loop sui seed
    results = []
    for env_seed in env_seeds:
        algo_seed = env_seed + args.algo_seed_offset
        print(f"--- seed env={env_seed}, algo={algo_seed} ---")

        result = run_single_seed(
            mode=args.mode,
            env_seed=env_seed,
            algo_seed=algo_seed,
            output_dir=output_dir,
            normalize_reward=args.normalize_reward,
            verbose=args.verbose,
        )
        results.append(result)

        print(f"  best_eval_reward:  {result['best_mean_reward_eval']:.2f}")
        print(f"  final_eval_reward: {result['final_mean_reward_eval']:.2f}")
        print(f"  training time:     {result['training_time_seconds']:.1f}s")
        print()

    # Sintesi aggregata su tutti i seed
    # Prodotta anche con un solo seed: runs_summary.json ha cosi'
    # struttura costante e semplifica gli script di analisi successivi
    
    best_rewards  = [r["best_mean_reward_eval"]  for r in results]
    final_rewards = [r["final_mean_reward_eval"] for r in results]
    train_times   = [r["training_time_seconds"]  for r in results]

    summary = {
        "mode":                args.mode,
        "normalize_reward":    args.normalize_reward,
        "n_runs":              len(results),
        "runs":                results,
        "aggregate": {
            "best_mean_reward":             float(np.mean(best_rewards)),
            "best_std_reward":              float(np.std(best_rewards)),
            "final_mean_reward":            float(np.mean(final_rewards)),
            "final_std_reward":             float(np.std(final_rewards)),
            "total_training_time_seconds":  float(np.sum(train_times)),
        },
    }
    summary_path = output_dir / args.mode / "runs_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Sintesi run salvata in: {summary_path}")
    print(f"  best_mean  (su {len(results)} seed): "
          f"{summary['aggregate']['best_mean_reward']:.2f} "
          f"± {summary['aggregate']['best_std_reward']:.2f}")
    print(f"  final_mean (su {len(results)} seed): "
          f"{summary['aggregate']['final_mean_reward']:.2f} "
          f"± {summary['aggregate']['final_std_reward']:.2f}")


if __name__ == "__main__":
    main()