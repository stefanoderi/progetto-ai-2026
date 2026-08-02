"""
tools/build_summary.py — costruisce synthesis.txt, la sintesi finale
del confronto baseline + DQN + PPO, rispondendo a tre domande:
    A) gli agenti RL battono le baseline?
    B) DQN e PPO si comportano in modo simile o diverso?
    C) il problema e' stato appreso o solo sopravvissuto?

Integra tre fonti gia' prodotte: comparison_final.txt (la tabella di
evaluate.py), i _summaries.jsonl (le 50 reward per episodio, per i bootstrap
appaiati) e i runs_summary.json (numeri di training su 5 seed). 
L'output cita la tabella, mostra i CI appaiati dei confronti chiave, 
distingue 'appreso' (reward molto sopra do_nothing, win rate alto, comp_val basso, azioni sensate) 
da sopravvissuto (reward poco sopra do_nothing), e chiude con conclusioni a tre voci 1:1 con le domande.

Uso:
    python tools/build_summary.py \\
        --comparison-table runs/final_comparison/comparison_final.txt \\
        --eval-dir         runs/final_comparison/evaluations \\
        --dqn-summary      runs/dqn/full/runs_summary.json \\
        --ppo-summary      runs/ppo/full/runs_summary.json \\
        --dqn-name dqn_best_model.zip_ --ppo-name ppo_best_model.zip_ \\
        --out              runs/final_comparison/synthesis.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np


# Helpers di I/O

def load_summary_rewards(jsonl_path: Path) -> np.ndarray:
    """Reward per episodio da un _summaries.jsonl (chiave total_reward)"""
    rewards = []
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if "total_reward" in rec:
                rewards.append(float(rec["total_reward"]))
    return np.asarray(rewards, dtype=np.float64)


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray,
                          n_boot: int = 10_000, seed: int = 12345) -> dict:
    """
    Bootstrap appaiato, stessa logica di bootstrap_ci.py ricopiata qui per non
    dipendere da quel modulo. Restituisce mean_delta, ci_low, ci_high, significant.
    """
    if len(a) != len(b):
        raise ValueError(f"len mismatch: {len(a)} vs {len(b)}")
    delta = a - b
    rng = np.random.default_rng(seed)
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = delta[idx].mean(axis=1)
    mean_delta = float(delta.mean())
    ci_low  = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    significant = (ci_low > 0) or (ci_high < 0)
    return {
        "mean_delta":  mean_delta,
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "significant": significant,
    }


def fmt_ci(d: dict) -> str:
    """Formatta un risultato di bootstrap in una riga compatta"""
    marker = "***" if d["significant"] else "n.s."
    return (f"Δ = {d['mean_delta']:+.2f}  "
            f"CI95 [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]  {marker}")


# Classificazione "appreso vs sopravvissuto"

def classify_learned_vs_survived(reward: float, do_nothing_reward: float,
                                 heuristic_reward: float,
                                 win_rate: float,
                                 invalid_rate: float) -> str:
    """
    Etichetta qualitativa. 
    - "appreso" se la policy ha reward >> do_nothing,  vince spesso (clean_streak target) 
    e fa poche azioni invalide; 
    - "parzialmente appreso" se soddisfa solo una o due delle tre; 
    - "solo sopravvissuto" se il vantaggio su do_nothing e' marginale. 
    
    Le soglie sono larghe di proposito: nel testo seguono i numeri esatti
    """
    margin_dn   = reward - do_nothing_reward       # vantaggio su do_nothing
    margin_heur = reward - heuristic_reward         # vantaggio sulla heuristic

    c_above_dn    = margin_dn > 30.0      # vantaggio netto su do_nothing
    c_winning     = win_rate > 0.30       # vince almeno 1 episodio su 3
    c_low_invalid = invalid_rate < 0.30   # non troppi sprechi

    n_true = sum([c_above_dn, c_winning, c_low_invalid])
    if margin_heur > 30.0:
        return "appreso (sopra anche la heuristic)"
    if n_true == 3:
        return "appreso"
    if n_true >= 2:
        return "parzialmente appreso"
    if margin_dn > 5.0:
        return "marginalmente meglio di do_nothing"
    return "solo sopravvissuto (≈ do_nothing)"


# Parsing della tabella di confronto di evaluate.py

def parse_comparison_table(txt_path: Path) -> dict:
    """
    Parsa comparison_final.txt (tabella ASCII a colonne separate da spazi):
    trova la riga di header che inizia con "policy", poi legge le righe di dati
    non vuote e non di separatore. Restituisce {policy_name: {metrica: float}}
    """
    lines = txt_path.read_text().splitlines()

    # Riga di header (inizia con "policy")
    header_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("policy"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Header 'policy' non trovato in {txt_path}")

    header_fields = lines[header_idx].split()
    out = {}

    # Dopo l'header c'e' una riga di trattini: la saltiamo e leggiamo i dati
    i = header_idx + 1
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("-") or not line.strip():
            i += 1
            continue
        fields = line.split()
        if len(fields) < len(header_fields):
            i += 1
            continue
        policy_name = fields[0]
        try:
            values = [float(x) for x in fields[1:len(header_fields)]]
        except ValueError:
            i += 1   # riga non parsabile -> non sono dati, skip
            continue
        out[policy_name] = dict(zip(header_fields[1:], values))
        i += 1
    return out


# Generazione della sintesi finale

def build_synthesis(table_path: Path, eval_dir: Path,
                    dqn_summary_path: Path, ppo_summary_path: Path,
                    dqn_jsonl_prefix: str, ppo_jsonl_prefix: str,
                    seed: int = 42) -> str:
    """
    Costruisce il testo completo di synthesis.txt. I prefissi JSONL individuano i
    _summaries.jsonl dei due agenti RL dentro eval_dir; il path viene composto
    come eval_<prefix>_seed<S>_summaries.jsonl.
    """
    out_lines = []

    # 0) Header
    out_lines.append("=" * 78)
    out_lines.append("Sintesi finale del confronto fra metodi")
    out_lines.append("=" * 78)
    out_lines.append("")

    # 1) Tabella numerica
    out_lines.append("1) Tabella numerica completa")
    out_lines.append("-" * 78)
    out_lines.append(table_path.read_text().rstrip())
    out_lines.append("")

    # 2) Parsing dei numeri: in tabella DQN e PPO hanno il nome esteso
    #    (es. "dqn[best_model.zip]"): cerchiamo quelli che iniziano con dqn[ o ppo[
    metrics = parse_comparison_table(table_path)
    dqn_name = next((k for k in metrics if k.startswith("dqn[")), None)
    ppo_name = next((k for k in metrics if k.startswith("ppo[")), None)
    if dqn_name is None or ppo_name is None:
        raise ValueError(
            f"Non sono riuscito a trovare dqn[...] e ppo[...] nella tabella. "
            f"Policy trovate: {list(metrics.keys())}"
        )

    # 3) Reward di riferimento
    do_nothing_r = metrics["do_nothing"]["reward_mean"]
    heuristic_r  = metrics["heuristic"]["reward_mean"]
    oracle_r     = metrics["oracle"]["reward_mean"]
    dqn_r        = metrics[dqn_name]["reward_mean"]
    ppo_r        = metrics[ppo_name]["reward_mean"]

    # 4) Confronti bootstrap appaiati
    out_lines.append("2) Confronti appaiati (bootstrap CI 95% sulla differenza)")
    out_lines.append("-" * 78)
    out_lines.append("Nota: ogni confronto usa gli stessi 50 scenari (stesso "
                     "seed condiviso fra")
    out_lines.append("policy). Δ = mean(A) - mean(B) con A=prima policy "
                     "menzionata. *** = ")
    out_lines.append("0 fuori dal CI95 al 5%; n.s. = differenza non "
                     "significativa.")
    out_lines.append("")

    def load_for(name: str) -> np.ndarray:
        """Risolve il path del jsonl per nome di policy della tabella."""
        if name == "heuristic":
            return load_summary_rewards(eval_dir / f"eval_heuristic_seed{seed}_summaries.jsonl")
        if name == "do_nothing":
            return load_summary_rewards(eval_dir / f"eval_do_nothing_seed{seed}_summaries.jsonl")
        if name == "oracle":
            return load_summary_rewards(eval_dir / f"eval_oracle_seed{seed}_summaries.jsonl")
        if name == "panic_isolation":
            return load_summary_rewards(eval_dir / f"eval_panic_isolation_seed{seed}_summaries.jsonl")
        if name == "random":
            return load_summary_rewards(eval_dir / f"eval_random_seed{seed}_summaries.jsonl")
        if name == "dqn":
            return load_summary_rewards(eval_dir / f"eval_{dqn_jsonl_prefix}_seed{seed}_summaries.jsonl")
        if name == "ppo":
            return load_summary_rewards(eval_dir / f"eval_{ppo_jsonl_prefix}_seed{seed}_summaries.jsonl")
        raise ValueError(f"name {name} non gestito")

    r_dqn       = load_for("dqn")
    r_ppo       = load_for("ppo")
    r_heur      = load_for("heuristic")
    r_donothing = load_for("do_nothing")

    confronti = [
        ("PPO",       "Heuristic",  r_ppo, r_heur),
        ("DQN",       "Heuristic",  r_dqn, r_heur),
        ("PPO",       "DQN",        r_ppo, r_dqn),
        ("PPO",       "DoNothing",  r_ppo, r_donothing),
        ("DQN",       "DoNothing",  r_dqn, r_donothing),
    ]
    for label_a, label_b, a, b in confronti:
        d = paired_bootstrap_diff(a, b)
        out_lines.append(f"  {label_a:<10} vs {label_b:<12}  {fmt_ci(d)}")

    out_lines.append("")

    # 5) Valutazione qualitativa per ciascun agente RL
    out_lines.append("3) Diagnosi qualitativa per ciascun agente RL")
    out_lines.append("-" * 78)
    for label, key in [("DQN", dqn_name), ("PPO", ppo_name)]:
        m = metrics[key]
        diag = classify_learned_vs_survived(
            reward=m["reward_mean"],
            do_nothing_reward=do_nothing_r,
            heuristic_reward=heuristic_r,
            win_rate=m["win%"] / 100.0,
            invalid_rate=m["inv%"] / 100.0,
        )
        out_lines.append(f"  {label} ({key}):")
        out_lines.append(f"     reward            = {m['reward_mean']:+.2f}")
        out_lines.append(f"     vs do_nothing     = {m['reward_mean']-do_nothing_r:+.2f}")
        out_lines.append(f"     vs heuristic      = {m['reward_mean']-heuristic_r:+.2f}")
        out_lines.append(f"     vs oracle (tetto) = {m['reward_mean']-oracle_r:+.2f}")
        out_lines.append(f"     win%              = {m['win%']:.1f}%")
        out_lines.append(f"     invalid%          = {m['inv%']:.1f}%")
        out_lines.append(f"     comp_val          = {m['comp_val']:.2f}")
        out_lines.append(f"     diagnosi          = {diag}")
        out_lines.append("")

    # 6) Stabilita' del training fra seed
    out_lines.append("4) Stabilita' del training fra seed (da runs_summary.json)")
    out_lines.append("-" * 78)
    for label, path in [("DQN", dqn_summary_path), ("PPO", ppo_summary_path)]:
        with path.open() as f:
            summary = json.load(f)
        agg = summary["aggregate"]
        out_lines.append(
            f"  {label}: best_eval mean = {agg['best_mean_reward']:+.2f} "
            f"± {agg['best_std_reward']:.2f}   "
            f"final_eval mean = {agg['final_mean_reward']:+.2f} "
            f"± {agg['final_std_reward']:.2f}"
        )
    # I valori qui sono nelle scale di TRAINING (DQN normalizzata, PPO naturale),
    # mentre la tabella della sezione 1 e' in scala naturale per entrambi
    out_lines.append("")
    out_lines.append("  Nota: i valori in questa sezione sono nelle scale di")
    out_lines.append("        TRAINING (DQN normalizzata in [-1,0], PPO naturale).")
    out_lines.append("        La tabella della sezione 1 e' invece in scala")
    out_lines.append("        naturale per entrambi gli algoritmi.")
    out_lines.append("")

    # 7) Conclusioni formali sulle tre domande di partenza
    out_lines.append("5) Conclusioni formali")
    out_lines.append("-" * 78)

    # A) RL battono le baseline?
    d_ppo_heur = paired_bootstrap_diff(r_ppo, r_heur)
    d_dqn_heur = paired_bootstrap_diff(r_dqn, r_heur)
    d_ppo_dn   = paired_bootstrap_diff(r_ppo, r_donothing)
    d_dqn_dn   = paired_bootstrap_diff(r_dqn, r_donothing)

    out_lines.append("")
    out_lines.append("A) Gli agenti RL battono le baseline?")
    if d_ppo_heur["significant"] and d_ppo_heur["mean_delta"] > 0:
        out_lines.append("   - PPO: SI, batte la heuristic in modo significativo "
                         f"(Δ = {d_ppo_heur['mean_delta']:+.2f}, CI95 "
                         f"[{d_ppo_heur['ci_low']:+.2f}, "
                         f"{d_ppo_heur['ci_high']:+.2f}]).")
    elif d_ppo_heur["significant"] and d_ppo_heur["mean_delta"] < 0:
        out_lines.append("   - PPO: NO, e' significativamente peggio della "
                         f"heuristic (Δ = {d_ppo_heur['mean_delta']:+.2f}).")
    else:
        out_lines.append("   - PPO: pareggio statistico con la heuristic "
                         f"(Δ = {d_ppo_heur['mean_delta']:+.2f}, "
                         "differenza non significativa).")

    if d_dqn_heur["significant"] and d_dqn_heur["mean_delta"] > 0:
        out_lines.append("   - DQN: SI, batte la heuristic in modo significativo "
                         f"(Δ = {d_dqn_heur['mean_delta']:+.2f}).")
    elif d_dqn_heur["significant"] and d_dqn_heur["mean_delta"] < 0:
        out_lines.append("   - DQN: NO, e' significativamente peggio della "
                         f"heuristic (Δ = {d_dqn_heur['mean_delta']:+.2f}, "
                         f"CI95 [{d_dqn_heur['ci_low']:+.2f}, "
                         f"{d_dqn_heur['ci_high']:+.2f}]).")
    else:
        out_lines.append("   - DQN: pareggio statistico con la heuristic "
                         f"(Δ = {d_dqn_heur['mean_delta']:+.2f}).")

    out_lines.append("   - Su do_nothing (limite inferiore di confronto):")
    out_lines.append(f"       PPO  vs  do_nothing : {fmt_ci(d_ppo_dn)}")
    out_lines.append(f"       DQN  vs  do_nothing : {fmt_ci(d_dqn_dn)}")
    out_lines.append("")

    # B) DQN e PPO si comportano in modo simile o diverso?
    d_ppo_dqn = paired_bootstrap_diff(r_ppo, r_dqn)
    out_lines.append("B) DQN e PPO si comportano in modo simile o diverso?")
    out_lines.append(f"     PPO  vs  DQN: {fmt_ci(d_ppo_dqn)}")
    if d_ppo_dqn["significant"]:
        winner, loser = ("PPO", "DQN") if d_ppo_dqn["mean_delta"] > 0 else ("DQN", "PPO")
        out_lines.append(f"     -> {winner} e' significativamente meglio "
                         f"di {loser}: i due algoritmi NON sono equivalenti.")
    else:
        out_lines.append("     -> i due algoritmi sono statisticamente "
                         "equivalenti sulla reward media.")
    # Distanza qualitativa fra le composizioni (win%, inv%, comp_val)
    out_lines.append("   Differenze qualitative:")
    for metric_key, pretty in [("win%", "win%"), ("inv%", "inv%"), ("comp_val", "comp_val")]:
        v_dqn = metrics[dqn_name][metric_key]
        v_ppo = metrics[ppo_name][metric_key]
        out_lines.append(f"     {pretty:<10}  DQN = {v_dqn:6.2f}   PPO = {v_ppo:6.2f}   "
                         f"(Δ = {v_ppo - v_dqn:+.2f})")
    out_lines.append("")

    # C) Appreso o sopravvissuto?
    out_lines.append("C) Il problema e' stato appreso o solo \"sopravvissuto\"?")
    for label, key in [("DQN", dqn_name), ("PPO", ppo_name)]:
        m = metrics[key]
        diag = classify_learned_vs_survived(
            reward=m["reward_mean"],
            do_nothing_reward=do_nothing_r,
            heuristic_reward=heuristic_r,
            win_rate=m["win%"] / 100.0,
            invalid_rate=m["inv%"] / 100.0,
        )
        out_lines.append(f"   - {label}: {diag}")
    out_lines.append("")
    out_lines.append("=" * 78)

    return "\n".join(out_lines)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera la sintesi testuale finale del confronto baseline + DQN + PPO."
    )
    p.add_argument("--comparison-table", type=Path, required=True,
                   help="path al comparison_final.txt prodotto da evaluate.py")
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="cartella contenente i _summaries.jsonl di ciascuna policy")
    p.add_argument("--dqn-summary", type=Path, required=True,
                   help="path al runs/dqn/full/runs_summary.json")
    p.add_argument("--ppo-summary", type=Path, required=True,
                   help="path al runs/ppo/full/runs_summary.json")
    p.add_argument("--dqn-name", type=str, default="dqn_best_model.zip_",
                   help="prefisso del jsonl di DQN dentro --eval-dir "
                        "(quello che sta fra 'eval_' e '_seed<N>'). Default: "
                        "'dqn_best_model.zip_'")
    p.add_argument("--ppo-name", type=str, default="ppo_best_model.zip_",
                   help="prefisso del jsonl di PPO dentro --eval-dir. Default: "
                        "'ppo_best_model.zip_'")
    p.add_argument("--seed", type=int, default=42,
                   help="seed di evaluate.py usato per i file _summaries.jsonl. Default: 42")
    p.add_argument("--out", type=Path,
                   default=Path("runs/final_comparison/synthesis.txt"),
                   help="path del file di sintesi da produrre. "
                        "Default: runs/final_comparison/synthesis.txt")
    return p


def main():
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = build_synthesis(
        table_path=args.comparison_table,
        eval_dir=args.eval_dir,
        dqn_summary_path=args.dqn_summary,
        ppo_summary_path=args.ppo_summary,
        dqn_jsonl_prefix=args.dqn_name,
        ppo_jsonl_prefix=args.ppo_name,
        seed=args.seed,
    )
    args.out.write_text(text)
    print(text)
    print()
    print(f"[ok] Sintesi scritta in: {args.out}")


if __name__ == "__main__":
    main()
