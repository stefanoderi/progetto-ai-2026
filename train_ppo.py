"""
train_ppo.py — addestramento di PPO su IncidentResponseEnv.

Tre modalita' via --mode, con budget crescenti (PPO e' on-policy: niente
learning_starts da regolare come in DQN):
    smoke        ~10k step: verifica end-to-end che la pipeline giri e la reward
                 si muova.
    preliminary  ~200k step: osserva andamento reward, calo delle azioni
                 invalide e stabilita' prima della run full.
    full         500k step per seed, iperparametri = default SB3; con
                 --seeds 0,1,2,3,4 produce 5 run nella stessa cartella.

Output per run: <out_dir>/<mode>/seed<S>/ con best_model/best_model.zip,
eval_logs/evaluations.npz, final_model.zip, meta.json, tb_logs/. Con piu' seed
viene prodotto anche runs_summary.json (aggregati best/final mean +/- std).

Seed: env_seed (patient_zero, propagazione, rumore) e algo_seed (init actor/
critic, sampling dei rollout, shuffle dei minibatch) sono separati e registrati
nei meta.json. Default algo_seed = env_seed + 1000.
Come per DQN, il train env riceve algo_seed al primo reset tramite
DummyVecEnv.seed(algo_seed), che sovrascrive l'env_seed di istanziazione.
Quindi env_seed di fatto non controlla il training (lo fa algo_seed); la
separazione regge solo per l'eval env, seedato a env_seed + 10_000.

Reward normalization (--normalize-reward): divide la reward per 17 (somma dei
business value); non cambia la policy ottimale. A differenza di
DQN, PPO normalizza internamente i vantaggi (normalize_advantage=True) e usa la
value function come baseline, quindi e' strutturalmente meno sensibile alla
scala della reward: va attivata solo in caso di divergenza/stagnazione.

Esempi:
    python train_ppo.py --mode smoke --seed 42
    python train_ppo.py --mode full --seeds 0,1,2,3,4
    python train_ppo.py --mode full --seeds 0,1,2,3,4 --normalize-reward
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.incident_response_env import IncidentResponseEnv


# Normalizzazione reward (opzionale)

# 17 = somma di tutti i business value (5*1 + 2*3 + 1*6): dividere per 17 porta
# la reward circa in [-1, 0] senza alterare la policy ottimale. Costante e
# wrapper sono duplicati da train_dqn.py cosi' i due file di training
# restano indipendenti.
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


# --- Iperparametri PPO (default Stable-Baselines3 v2.x) ---
@dataclass
class PPOHparams:
    """
    Iperparametri di PPO, tutti default di SB3 v2.x. PPO non ha learning_starts
    (on-policy). 
    Default piu' rilevanti:
    - n_steps=2048: step di rollout prima di un update. Con episodi da max 40
      step un rollout copre ~50 episodi
    - normalize_advantage=True (default SB3): PPO normalizza i vantaggi, quindi
      e' meno sensibile alla scala della reward rispetto a DQN
    - clip_range=0.2: clipping del rapporto policy nuova/vecchia
    - ent_coef=0.0: nessun bonus di entropia esplicito
    """
    # Apprendimento
    learning_rate: float = 3e-4
    n_steps:       int   = 2048
    batch_size:    int   = 64
    n_epochs:      int   = 10
    gamma:         float = 0.99

    # Vantaggio
    gae_lambda:    float = 0.95

    # Clipping
    clip_range:    float = 0.2

    # Regolarizzazione
    ent_coef:      float = 0.0
    vf_coef:       float = 0.5
    max_grad_norm: float = 0.5


# --- Configurazione delle modalita' ---
# Stessi budget di timesteps di train_dqn.py . 
# Le eval_freq sono allineate a DQN; PPO le arrotonda internamente all'inizio del rollout successivo
MODE_DEFAULTS = {
    "smoke": {
        "total_timesteps":  10_000,
        "eval_freq":         2_000,
        "n_eval_episodes":      10,
    },
    "preliminary": {
        "total_timesteps": 200_000,
        "eval_freq":        10_000,
        "n_eval_episodes":      20,
    },
    "full": {
        "total_timesteps": 500_000,
        "eval_freq":        25_000,
        "n_eval_episodes":      50,
    },
}


# --- Costruzione dell'ambiente ---
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


# --- Helper interno ---
def _get_sb3_version() -> str:
    """Versione di stable-baselines3 (registrata in meta.json)."""
    try:
        import stable_baselines3 as sb3
        return sb3.__version__
    except Exception:
        return "unknown"


# --- Esecuzione di un singolo training con un singolo seed ---
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
    best_mean_reward_eval, final_mean_reward_eval, run_dir.
    """
    cfg = MODE_DEFAULTS[mode]
    hparams = PPOHparams()

    # Directory di output
    run_dir  = output_dir / mode / f"seed{env_seed}"
    tb_dir   = run_dir / "tb_logs"
    eval_dir = run_dir / "eval_logs"
    best_dir = run_dir / "best_model"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Eval env con seed offset +10000 rispetto al train env: gli scenari
    # dell'eval periodica sono diversi da quelli del rollout di training,
    # cosi' si evita data leakage nella valutazione interna.
    env_train = make_env(seed=env_seed,
                         normalize_reward=normalize_reward)
    env_eval  = make_env(seed=env_seed + 10_000,
                         normalize_reward=normalize_reward)

    # Callback di eval periodica: salva il best model (miglior reward medio) in
    # best_dir/best_model.zip e la storia dei reward in eval_dir/evaluations.npz.
    # Su PPO l'eval_freq e' "snapped" all'inizio del rollout successivo (multipli
    # di n_steps=2048): comportamento standard di SB3, non un bug.
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

    # ---- Modello PPO ----
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=hparams.learning_rate,
        n_steps=hparams.n_steps,
        batch_size=hparams.batch_size,
        n_epochs=hparams.n_epochs,
        gamma=hparams.gamma,
        gae_lambda=hparams.gae_lambda,
        clip_range=hparams.clip_range,
        ent_coef=hparams.ent_coef,
        vf_coef=hparams.vf_coef,
        max_grad_norm=hparams.max_grad_norm,
        seed=algo_seed,
        tensorboard_log=str(tb_dir),
        verbose=verbose,
    )

    # ---- Training ----
    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=eval_callback,
        progress_bar=False,   # off
        log_interval=10,
    )
    t_elapsed = time.time() - t0

    # ---- Salvataggio modello finale ----
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

    # ---- meta.json: tutto cio' che serve a riprodurre la run ----
    meta = {
        "algorithm":                  "PPO",
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
        description="Addestra PPO su IncidentResponseEnv.",
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

    p.add_argument("--output-dir", type=str, default="runs/ppo",
                   help="directory radice di output (default: 'runs/ppo')")

    p.add_argument("--normalize-reward", action="store_true",
                   help="divide la reward per 17 (somma BV) per migliorare "
                        "la stabilita' numerica. PPO normalizza gia' i vantaggi "
                        "internamente; questa flag e' qui per simmetria con DQN.")

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

    print(f"=== PPO training — mode={args.mode}, seeds={env_seeds} ===")
    print(f"output:           {output_dir / args.mode}")
    print(f"normalize_reward: {args.normalize_reward}")
    print(f"sb3 version:      {_get_sb3_version()}")
    print()

    # ---- Loop sui seed ----
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

    # ---- Sintesi aggregata su tutti i seed ----
    # Prodotta sempre, anche con un solo seed: runs_summary.json ha cosi'
    # struttura costante e semplifica gli script di analisi successivi.
    best_rewards  = [r["best_mean_reward_eval"]  for r in results]
    final_rewards = [r["final_mean_reward_eval"] for r in results]
    train_times   = [r["training_time_seconds"]  for r in results]

    summary = {
        "algorithm":           "PPO",
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