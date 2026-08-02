"""
tools/bootstrap_ci.py — intervallo di confidenza bootstrap appaiato sulla
differenza di reward media fra due policy, leggendo i _summaries.jsonl di
evaluate.py.

Appaiato e non indipendente: a --seed fissato, le N=50 evaluation delle due
policy condividono gli stessi episode_seed, quindi l'episodio i di A e l'episodio
i di B partono dallo stesso scenario. La varianza dovuta allo scenario si
cancella nella differenza r_A_i - r_B_i, per questo si usa il test appaiato. 
L'ordine degli episodi quindi non va cambiato.

Procedura: costruisce delta_i = r_A_i - r_B_i (i due file devono avere lo stesso
N), poi bootstrap (default 10_000 iterazioni) ricampionando N indici con
reinserimento e mediando i delta. Riporta mean(delta), CI95 dai percentili
2.5/97.5 e un "p-value bilaterale".

# Il "p-value" riportato è solo indicativo, il riferimento è il CI95: 
# se non contiene lo zero, la differenza è significativa al 5%.

Uso:
    python tools/bootstrap_ci.py --a <A_summaries.jsonl> --b <B_summaries.jsonl> \\
        --label-a "PPO" --label-b "Heuristic" --n-boot 10000 --seed 12345 \\
        > runs/final_comparison/significance/ppo_vs_heuristic.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_rewards(jsonl_path: Path) -> np.ndarray:
    """
    Legge un _summaries.jsonl e restituisce il vettore di reward per episodio
    nell'ordine di esecuzione (= stessi episode_seed): il bootstrap appaiato
    accoppia l'episodio i di A con l'episodio i di B, quindi non va riordinato.
    """
    rewards = []
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            # chiave canonica prodotta da utils.summarize_episode; fail-loud se assente
            if "total_reward" not in rec:
                raise KeyError(
                    f"chiave 'total_reward' assente nel record: "
                    f"{list(rec.keys())} (file: {jsonl_path})"
                )
            rewards.append(float(rec["total_reward"]))
    return np.asarray(rewards, dtype=np.float64)


def paired_bootstrap_ci(delta: np.ndarray, n_boot: int = 10_000,
                        seed: int = 12345,
                        alpha: float = 0.05) -> dict:
    """
    Bootstrap appaiato sulla media dei delta. Restituisce mean_delta, ci_low,
    ci_high (CI a 1-alpha), p_two_sided (proxy), n, n_boot.
    """
    n = len(delta)
    rng = np.random.default_rng(seed)

    # Vettorizzato: matrice (n_boot, N) di indici ricampionati -> media per riga
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = delta[idx].mean(axis=1)

    mean_delta = float(delta.mean())
    ci_low  = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # p-value bilaterale: frazione di bootstrap col segno opposto a quello
    # osservato, raddoppiata e clippata a 1
    if mean_delta >= 0:
        p_one_side = (boot_means <= 0).mean()
    else:
        p_one_side = (boot_means >= 0).mean()
    p_two_sided = float(min(1.0, 2.0 * p_one_side))

    return {
        "mean_delta":   mean_delta,
        "ci_low":       ci_low,
        "ci_high":      ci_high,
        "p_two_sided":  p_two_sided,
        "n":            n,
        "n_boot":       n_boot,
    }


def format_report(result: dict, label_a: str, label_b: str,
                  rewards_a: np.ndarray, rewards_b: np.ndarray) -> str:
    """Formatta il risultato in un report multilinea: descrittive, bootstrap, interpretazione."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"Bootstrap CI appaiato: {label_a}  vs  {label_b}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Statistiche descrittive (su n_episodi):")
    lines.append(f"  {label_a:<30} mean = {rewards_a.mean():+8.3f}  std = {rewards_a.std():7.3f}")
    lines.append(f"  {label_b:<30} mean = {rewards_b.mean():+8.3f}  std = {rewards_b.std():7.3f}")
    lines.append(f"  numero episodi appaiati:        {result['n']}")
    lines.append("")
    lines.append("Bootstrap sui delta appaiati (delta = A - B):")
    lines.append(f"  mean(delta)            = {result['mean_delta']:+.3f}")
    lines.append(f"  CI 95% per mean(delta) = [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]")
    lines.append(f"  p-value bilaterale     = {result['p_two_sided']:.4f}")
    lines.append(f"  n_bootstrap            = {result['n_boot']}")
    lines.append("")

    # Se 0 e' fuori dal CI95, differenza significativa al 5%
    significant = (result['ci_low'] > 0) or (result['ci_high'] < 0)
    if significant:
        direction = (
            f"{label_a} BATTE {label_b}"
            if result['mean_delta'] > 0
            else f"{label_b} BATTE {label_a}"
        )
        lines.append(f"Interpretazione: 0 NON e' nel CI95 -> differenza "
                     f"significativa al 5%.")
        lines.append(f"                 {direction} (vantaggio medio "
                     f"= {abs(result['mean_delta']):.3f} punti di reward).")
    else:
        lines.append("Interpretazione: 0 E' nel CI95 -> nessuna evidenza "
                     "di differenza significativa al 5%.")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bootstrap appaiato sulla differenza di reward fra due policy.",
    )
    p.add_argument("--a", type=Path, required=True,
                   help="path al _summaries.jsonl della policy A")
    p.add_argument("--b", type=Path, required=True,
                   help="path al _summaries.jsonl della policy B")
    p.add_argument("--label-a", type=str, default="A",
                   help="etichetta leggibile per la policy A")
    p.add_argument("--label-b", type=str, default="B",
                   help="etichetta leggibile per la policy B")
    p.add_argument("--n-boot", type=int, default=10_000,
                   help="numero di iterazioni bootstrap (default: 10000)")
    p.add_argument("--seed", type=int, default=12345,
                   help="seed del generatore bootstrap (default: 12345)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="livello di significativita', default 0.05 (-> CI95)")
    return p


def main():
    args = build_parser().parse_args()

    rewards_a = load_rewards(args.a)
    rewards_b = load_rewards(args.b)

    if len(rewards_a) != len(rewards_b):
        raise SystemExit(
            f"Errore: i due file hanno un numero diverso di episodi "
            f"({len(rewards_a)} vs {len(rewards_b)}). "
            f"Il bootstrap APPAIATO richiede stesso N e stesso ordine "
            f"di esecuzione (= stesso --seed di evaluate.py)."
        )

    delta = rewards_a - rewards_b
    result = paired_bootstrap_ci(delta, n_boot=args.n_boot,
                                 seed=args.seed, alpha=args.alpha)

    report = format_report(result, args.label_a, args.label_b,
                           rewards_a, rewards_b)
    print(report)


if __name__ == "__main__":
    main()
