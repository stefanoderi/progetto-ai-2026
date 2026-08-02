"""
tools/extract_action_dist.py — distribuzione delle azioni dai JSONL di
evaluation, per seed.

Per ogni cartella seedN sotto --root somma le occorrenze di ogni tipo di azione
leggendo i campi info di ciascuno step. 
Classifica ogni step in una delle sei categorie, 'invalid' se rifiutata (in classify_step)

Uso:
    python tools/extract_action_dist.py [--root <path>] [--prefix <pat>]
    default: root=runs/ppo/evaluations/full, prefix=eval_ppo
"""
import argparse
import json
from collections import Counter
from pathlib import Path


# I sei tipi di azione riportati in tabella
ACTION_TYPES = ["analyse", "isolate", "restore", "reconnect",
                "do_nothing", "invalid"]


def classify_step(info: dict) -> str:
    """
    "invalid" se l'azione e' stata rifiutata (oscura il tipo originale), 
    altrimenti action_type_original.
    """
    if info.get("action_was_invalid", False):
        return "invalid"
    return info.get("action_type_original", "OTHER") # fallback se il campo manca (JSONL vecchi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("runs/ppo/evaluations/full"),
                    help="directory che contiene le sottocartelle seedN")
    ap.add_argument("--prefix", type=str, default="eval_ppo",
                    help="prefisso dei file JSONL da scandire")
    args = ap.parse_args()

    seed_dirs = sorted(args.root.glob("seed*"))
    if not seed_dirs:
        print(f"Nessuna cartella seed* trovata sotto {args.root}")
        return

    # Intestazione a colonne fisse: seed, un tipo per colonna, total
    col_w = 11
    header = f"{'seed':<6}"
    for at in ACTION_TYPES:
        header += f"{at:<{col_w}}"
    header += f"{'total':<7}"
    print(header)
    print("-" * len(header))

    other_examples = Counter()

    for seed_dir in seed_dirs:
        seed_id = seed_dir.name.replace("seed", "")
        jsonls = sorted(seed_dir.glob(f"{args.prefix}*_steps.jsonl"))
        if not jsonls:
            print(f"{seed_id:<6} -- nessun file matching in {seed_dir} --")
            continue

        counter = Counter()
        for jsonl_path in jsonls:
            with jsonl_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    a_type = classify_step(rec["info"])
                    counter[a_type] += 1
                    if a_type == "OTHER":
                        other_examples[json.dumps(rec["info"])[:80]] += 1

        total = sum(counter.values())
        row = f"{seed_id:<6}"
        for at in ACTION_TYPES:
            row += f"{counter.get(at, 0):<{col_w}}"
        row += f"{total:<7}"
        print(row)

    # segnala step con info non classificato (JSONL di altre versioni)
    if other_examples:
        print()
        print("ATTENZIONE: step con info non classificato (esempi):")
        for s, n in other_examples.most_common(5):
            print(f"  {n:>6}x  {s}")


if __name__ == "__main__":
    main()
