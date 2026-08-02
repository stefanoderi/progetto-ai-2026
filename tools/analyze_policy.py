"""
tools/analyze_policy.py — analisi del comportamento di una policy appresa.

Carica uno o due modelli Stable-Baselines3, esegue N episodi sull'ambiente raccogliendo lo
stato reale a ogni passo, e produce cinque blocchi di analisi:

  1) distribuzione delle azioni
  2) comportamento condizionato
  3) scenari controllati
  4) confronto con le baseline 
  5) episodi rappresentativi

Il report principale viene stampato su stdout: va rediretto su file per conservarlo. Gli artefatti piu' voluminosi
o strutturati vengono invece scritti direttamente dallo script sotto
--out-dir:

  - representative_episodes_<tag>.txt : log passo-passo dei tre episodi scelti
  - key_numbers.json                  : numeri chiave in forma rileggibile

Uso (dalla root del progetto):

    mkdir -p runs/policy_analysis
    python tools/analyze_policy.py \\
        --ppo-model-path runs/ppo/full/seed3/best_model/best_model.zip \\
        --dqn-model-path runs/dqn/full/seed1/best_model/best_model.zip \\
        --seed 42 --n-episodes 50 --no-color \\
        > runs/policy_analysis/policy_analysis.txt

Nota: con --seed 42 --n-episodes 50 la sequenza di episodi e'
identica a quella usata dalla tabella di confronto gia' salvata (stesso seeding), 
quindi i totali di azione del blocco A coincidono esattamente con quelli.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Lo script sta in tools/ ma importa dalla root del progetto: la aggiungo al path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.incident_response_env import IncidentResponseEnv          # noqa: E402
from envs.topology import (                                          # noqa: E402
    build_network, NodeRuntime,
    NODE_TYPE_USER, NODE_TYPE_SERVICE, NODE_TYPE_CRITICAL,
)
from utils import (                                                  # noqa: E402
    action_to_str, state_legend,
    format_state, format_step_log, summarize_episode,
)
from baselines import (                                              # noqa: E402
    SB3PolicyWrapper,
    HeuristicDefenderPolicy,
    OracleDefenderPolicy,
    ACTION_DO_NOTHING,
)

_NETWORK        = build_network()
NODE_TYPE       = {s.node_id: s.node_type for s in _NETWORK}
CRITICAL_ID     = next(s.node_id for s in _NETWORK
                       if s.node_type == NODE_TYPE_CRITICAL)

ACTION_CATS = ["analyse", "isolate", "restore", "reconnect",
               "do_nothing", "invalid"]

# Tipi di azione che hanno un nodo bersaglio (escludono do_nothing).
TARGETED = {"analyse", "isolate", "restore", "reconnect"}


# Caricamento delle policy RL

def load_rl_policy(kind: str, path: str) -> SB3PolicyWrapper:
    """
    Carica un modello SB3 (DQN o PPO) e lo incapsula come policy.
    deterministic=True: la stessa osservazione da' sempre la stessa azione
    (necessario per il blocco C degli scenari controllati).
    """
    if kind == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(path)
    elif kind == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(path)
    else:
        raise ValueError(f"kind sconosciuto: {kind}")
    return SB3PolicyWrapper(model, deterministic=True, name=kind)


# Esecuzione di un episodio "ricco"

def run_episode_rich(env: IncidentResponseEnv, policy, episode_seed: int,
                     ref_policy=None) -> dict:
    """
    Come il run_episode dell'orchestratore, ma cattura a ogni passo anche:

      - obs_before        : l'osservazione su cui la policy ha deciso
      - true_comp_before  : stato reale di compromissione pre-azione (8 bool),
                            letto direttamente dall'env (cio' che l'agente non vede).
      - ref_action        : cosa avrebbe scelto ref_policy sulla stessa obs
                            (usato per il tasso di accordo con una baseline)

    Restituisce un dict con history, reward totale, flag di fine e la sintesi
    prodotta dagli stessi helper usati altrove (numeri confrontabili).
    """
    obs, info = env.reset(seed=episode_seed)
    policy.reset()
    if ref_policy is not None:
        ref_policy.reset()

    history = []
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        obs_before  = obs
        info_before = info
        # Fotografia dello stato reale PRIMA dell'azione
        true_comp_before = [rt.compromised for rt in env.runtime]

        action = policy.select_action(obs_before, info_before)

        # La ref_policy decide sulla STESSA obs che la policy analizzata ha incontrato
        ref_action = (ref_policy.select_action(obs_before, info_before)
                      if ref_policy is not None else None)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        history.append({
            "t":                info["current_step"],
            "true_comp_before": true_comp_before,
            "action":           action,
            "reward":           float(reward),
            "info":             dict(info),          # copia difensiva
            "ref_action":       ref_action,
            "state_snapshot":   format_state(env, use_color=False),
        })

    info_hist = [h["info"] for h in history]
    summary = summarize_episode(info_hist, total_reward,
                               terminated, truncated, env)
    return {
        "history":      history,
        "total_reward": total_reward,
        "terminated":   terminated,
        "truncated":    truncated,
        "summary":      summary,
    }


def run_rollouts(env, policy, seed: int, n_episodes: int, ref_policy=None):
    """
    Esegue n_episodes episodi derivando i seed da un master RNG seedato con
    `seed` (stesso schema dell'orchestratore).
    """
    master_rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        episode_seed = int(master_rng.integers(0, 2**31 - 1))
        episodes.append(
            run_episode_rich(env, policy, episode_seed, ref_policy=ref_policy)
        )
    return episodes


# Utilities di formattazione tabelle

def _table(headers, rows) -> str:
    """Tabella ASCII a colonne allineate, stesso stile della tabella di confronto."""
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cols) for i in range(len(headers))]

    def fmt(row):
        return "  ".join(str(x).ljust(w) for x, w in zip(row, widths))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt([str(c) for c in r]) for r in rows)
    return "\n".join(lines)


def _cat_of_step(info: dict) -> str:
    """Categoria canonica di un passo: 'invalid' se rifiutata, altrimenti il tipo."""
    if info.get("action_was_invalid", False):
        return "invalid"
    return info.get("action_type_original", "OTHER")


# A) Distribuzione delle azioni
def block_action_distribution(episodes) -> tuple[str, dict]:
    """
    Frequenza assoluta e relativa di ogni categoria di azione; ripartizione delle
    azioni con bersaglio per tipo di nodo; per le azioni invalide, il tipo di azione
    tentata dall'agente che è stata rifiutata.

    Restituisce (testo_report, numeri_chiave) — i numeri chiave finiscono anche
    nel JSON rileggibile.
    """
    cat_counts   = Counter()
    target_type  = Counter()        # tipo nodo bersaglio, solo azioni valide con target
    invalid_type = Counter()        # tipo originale delle invalide
    invalid_node = Counter()        # tipo nodo bersaglio delle invalide
    total_steps  = 0

    for ep in episodes:
        for h in ep["history"]:
            info = h["info"]
            total_steps += 1
            cat = _cat_of_step(info)
            cat_counts[cat] += 1

            atype = info.get("action_type_original")
            tgt   = info.get("action_target")

            if not info.get("action_was_invalid", False):
                if atype in TARGETED and tgt is not None:
                    target_type[NODE_TYPE[tgt]] += 1
            else:
                invalid_type[atype] += 1
                if tgt is not None:
                    invalid_node[NODE_TYPE[tgt]] += 1

    # Tabella categorie: conteggio + percentuale sul totale passi.
    rows = []
    for cat in ACTION_CATS:
        c = cat_counts.get(cat, 0)
        pct = 100.0 * c / total_steps if total_steps else 0.0
        rows.append([cat, c, f"{pct:.1f}%"])
    rows.append(["TOTALE", total_steps, "100.0%"])
    tbl_cat = _table(["categoria", "conteggio", "quota"], rows)

    # Ripartizione azioni-con-bersaglio per tipo di nodo.
    tgt_total = sum(target_type.values())
    tgt_rows = []
    for nt in (NODE_TYPE_USER, NODE_TYPE_SERVICE, NODE_TYPE_CRITICAL):
        c = target_type.get(nt, 0)
        pct = 100.0 * c / tgt_total if tgt_total else 0.0
        tgt_rows.append([nt, c, f"{pct:.1f}%"])
    tbl_tgt = _table(["tipo nodo bersaglio", "conteggio", "quota"], tgt_rows)

    # Dettaglio delle invalide (per capire che tipo di errore fa la policy).
    inv_total = cat_counts.get("invalid", 0)
    inv_rows = []
    for atype, c in invalid_type.most_common():
        pct = 100.0 * c / inv_total if inv_total else 0.0
        inv_rows.append([atype, c, f"{pct:.1f}%"])
    tbl_inv = _table(["tipo originale invalida", "conteggio", "quota"],
                     inv_rows) if inv_rows else "(nessuna azione invalida)"

    inv_node_rows = [[nt, invalid_node.get(nt, 0)]
                     for nt in (NODE_TYPE_USER, NODE_TYPE_SERVICE, NODE_TYPE_CRITICAL)]
    tbl_inv_node = _table(["tipo nodo bersaglio invalide", "conteggio"],
                          inv_node_rows)

    text = "\n".join([
        "Distribuzione delle azioni (aggregata su tutti gli episodi)",
        "",
        tbl_cat,
        "",
        "Azioni con bersaglio, ripartite per tipo di nodo:",
        tbl_tgt,
        "",
        "Dettaglio delle sole azioni invalide (tipo originale decodificato):",
        tbl_inv,
        "",
        tbl_inv_node,
    ])

    key = {
        "total_steps":         total_steps,
        "category_counts":     dict(cat_counts),
        "invalid_rate":        (inv_total / total_steps) if total_steps else 0.0,
        "target_type_counts":  dict(target_type),
        "invalid_type_counts": dict(invalid_type),
    }
    return text, key


# B) Comportamento condizionato allo stato
def block_conditional(episodes) -> tuple[str, dict]:
    """
    Tre domande comportamentali, tutte misurate sui rollout reali:

      B1  quanto spesso un intervento (Isolate/Restore) su un nodo e' preceduto
          da un Analyse sullo stesso nodo?

      B2  quanto spesso un nodo isolato ma sano viene poi
          ricollegato?

      B3  con che rapidita' e con quale tipo di azione la policy affronta un
          nodo appena compromesso, separando il nodo critico dagli altri?
    """
    # --- B1: intervento informato vs alla cieca ---
    informed = {"isolate": 0, "restore": 0}
    blind    = {"isolate": 0, "restore": 0}
    for ep in episodes:
        last_an = {}          # ultimo Analyse valido per nodo (indice di passo)
        last_re = defaultdict(lambda: -1)   # ultimo Restore per nodo
        for idx, h in enumerate(ep["history"]):
            info = h["info"]
            inv  = info.get("action_was_invalid", False)
            atype = info.get("action_type_original")
            tgt   = info.get("action_target")
            if not inv and atype in ("isolate", "restore") and tgt is not None:
                an = last_an.get(tgt)
                # informed = c'è stato un Analyse su questo nodo dopo l'ultimo Restore
                # (che azzera la conoscenza); altrimenti l'intervento è alla cieca.
                if an is not None and an > last_re[tgt]:
                    informed[atype] += 1
                else:
                    blind[atype] += 1
            # Aggiorna i tracker DOPO aver classificato il passo corrente.
            if not inv and atype == "analyse" and tgt is not None:
                last_an[tgt] = idx
            if not inv and atype == "restore" and tgt is not None:
                last_re[tgt] = idx

    def _rate(a, b):
        tot = a + b
        return (100.0 * a / tot) if tot else None

    b1_rows = []
    for act in ("isolate", "restore"):
        tot = informed[act] + blind[act]
        r = _rate(informed[act], blind[act])
        rate_str = f"{r:.1f}%" if r is not None else "n/a"
        b1_rows.append([act, informed[act], blind[act], tot, rate_str])
    tbl_b1 = _table(
        ["intervento", "preceduto da Analyse", "alla cieca", "totale", "quota informata"],
        b1_rows,
    )

    # --- B2: reconnect dopo isolamento di nodo sano ---
    total_healthy_iso = 0
    total_reconnected = 0
    for ep in episodes:
        healthy_iso = {}       # nodo -> passo del primo isolamento "da sano"
        reconnected = set()
        for idx, h in enumerate(ep["history"]):
            info = h["info"]
            inv  = info.get("action_was_invalid", False)
            atype = info.get("action_type_original")
            tgt   = info.get("action_target")
            if not inv and atype == "isolate" and tgt is not None:
                if not h["true_comp_before"][tgt]:      # davvero sano al momento
                    healthy_iso.setdefault(tgt, idx)
            if not inv and atype == "reconnect" and tgt is not None:
                if tgt in healthy_iso and idx > healthy_iso[tgt]:
                    reconnected.add(tgt)
        total_healthy_iso += len(healthy_iso)
        total_reconnected += len(reconnected)

    if total_healthy_iso > 0:
        b2_line = (f"Nodi isolati mentre sani: {total_healthy_iso}; "
                   f"di questi poi ricollegati: {total_reconnected} "
                   f"({100.0*total_reconnected/total_healthy_iso:.1f}%).")
    else:
        b2_line = ("Nessun nodo e' mai stato isolato mentre era sano "
                   "(la policy non usa l'isolamento come contenimento, "
                   "oppure isola solo nodi realmente compromessi).")

    # --- B3: reazione al primo compromesso, per tipo di nodo ---
    lat_by_type   = defaultdict(list)     # latenze (in passi) fino al primo intervento
    firstact_type = defaultdict(Counter)  # tipo del primo intervento risolutivo
    unaddr_by_type = Counter()            # compromessi mai affrontati
    seen_by_type   = Counter()            # totale eventi di compromissione
    for ep in episodes:
        hist = ep["history"]
        first_comp = {}
        for idx, h in enumerate(hist):
            for node, comp in enumerate(h["true_comp_before"]):
                if comp and node not in first_comp:
                    first_comp[node] = idx
        first_addr = {}
        for idx, h in enumerate(hist):
            info = h["info"]
            if info.get("action_was_invalid", False):
                continue
            atype = info.get("action_type_original")
            tgt   = info.get("action_target")
            if atype in ("isolate", "restore") and tgt is not None:
                if (tgt in first_comp and idx >= first_comp[tgt]
                        and tgt not in first_addr):
                    first_addr[tgt] = (idx, atype)
        for node, ci in first_comp.items():
            nt = NODE_TYPE[node]
            seen_by_type[nt] += 1
            if node in first_addr:
                ai, atype = first_addr[node]
                lat_by_type[nt].append(ai - ci)
                firstact_type[nt][atype] += 1
            else:
                unaddr_by_type[nt] += 1

    b3_rows = []
    for nt in (NODE_TYPE_USER, NODE_TYPE_SERVICE, NODE_TYPE_CRITICAL):
        seen = seen_by_type.get(nt, 0)
        lats = lat_by_type.get(nt, [])
        mean_lat = f"{np.mean(lats):.2f}" if lats else "n/a"
        addressed = len(lats)
        addr_pct = f"{100.0*addressed/seen:.1f}%" if seen else "n/a"
        fa = firstact_type.get(nt, Counter())
        fa_str = ", ".join(f"{k}={v}" for k, v in fa.most_common()) or "-"
        b3_rows.append([nt, seen, addr_pct, mean_lat, fa_str])
    tbl_b3 = _table(
        ["tipo nodo", "compromissioni", "affrontate", "latenza media", "primo intervento"],
        b3_rows,
    )

    text = "\n".join([
        "Comportamento condizionato allo stato",
        "",
        "B1 — L'intervento e' preceduto da un Analyse sullo stesso nodo?",
        "     (misura se la policy raccoglie informazione prima di agire)",
        tbl_b1,
        "",
        "B2 — Il contenimento di un nodo sano viene poi corretto con Reconnect?",
        "     " + b2_line,
        "",
        "B3 — Reazione al primo compromesso, distinta per tipo di nodo",
        "     (latenza = passi fra il compromesso reale e il primo Isolate/Restore)",
        tbl_b3,
    ])

    key = {
        "informed_intervention": informed,
        "blind_intervention":    blind,
        "healthy_isolations":    total_healthy_iso,
        "healthy_reconnected":   total_reconnected,
        "reaction_seen_by_type":  dict(seen_by_type),
        "reaction_mean_latency": {nt: (float(np.mean(v)) if v else None)
                                  for nt, v in lat_by_type.items()},
        "reaction_unaddressed":  dict(unaddr_by_type),
    }
    return text, key


# C) Scenari controllati a stato noto

def _make_probe(env, setter):
    """
    Porta l'env in uno stato completamente controllato: azzera il runtime
    (tutti i nodi sani, non isolati, non in restore, senza alert e senza
    conoscenza) e poi applica `setter` che imposta solo i campi voluti.
    Restituisce (obs, info, snapshot) senza far avanzare il tempo: la
    risposta di ciascuna policy e' la sua reazione deterministica a quello
    stato esatto.
    """
    env.reset(seed=0)
    env.runtime = [NodeRuntime() for _ in range(env.cfg.n_nodes)]
    setter(env.runtime)
    info = env._get_info(action_was_invalid=False, action_type="probe",
                         action_target=None, action_type_original=None)
    obs = env._get_obs()
    snapshot = format_state(env, use_color=False)
    return obs, info, snapshot


def _probe_catalog():
    """
    Insieme di stati costruiti a mano per isolare l'effetto di singole
    variabili (tipo di nodo, presenza di alert, conoscenza pregressa,
    isolamento). Ogni voce e' (nome, descrizione, setter).

    Convenzioni dei setter: rt e' la lista di NodeRuntime (uno per nodo).
    Nodi 0..4 = user (BV 1), 5..6 = service (BV 3), 7 = critical (BV 6).
    Il nodo 4 e' l'utente piu' centrale (adiacente a entrambi i service).
    """
    def s_alert_user(rt):
        rt[4].compromised = True; rt[4].alert_active = True         # user con alert, ignoto
    def s_known_user(rt):
        rt[4].compromised = True; rt[4].analysis_state = 2          # user noto compromesso
    def s_alert_critical(rt):
        rt[CRITICAL_ID].compromised = True; rt[CRITICAL_ID].alert_active = True
    def s_known_critical(rt):
        rt[CRITICAL_ID].compromised = True; rt[CRITICAL_ID].analysis_state = 2
    def s_known_periph_user(rt):
        rt[0].compromised = True; rt[0].analysis_state = 2          # user periferico noto
    def s_known_service(rt):
        rt[5].compromised = True; rt[5].analysis_state = 2          # service noto compromesso
    def s_isolated_clean(rt):
        rt[0].isolated = True; rt[0].analysis_state = 1             # isolato e noto sano
    def s_restoring_only(rt):
        rt[3].isolated = True; rt[3].restore_timer = 2              # un nodo in restore, resto pulito
    def s_user_and_critical(rt):
        rt[0].compromised = True; rt[0].analysis_state = 2
        rt[CRITICAL_ID].compromised = True; rt[CRITICAL_ID].analysis_state = 2
    def s_false_alert(rt):
        rt[1].alert_active = True                                   # alert su nodo SANO, ignoto

    return [
        ("user_alert_ignoto",     "user centrale con alert, stato ignoto",          s_alert_user),
        ("user_noto_comp",        "user centrale noto compromesso, non isolato",    s_known_user),
        ("critical_alert_ignoto", "critico con alert, stato ignoto",                s_alert_critical),
        ("critical_noto_comp",    "critico noto compromesso, non isolato",          s_known_critical),
        ("user_perif_noto_comp",  "user periferico (nodo 0) noto compromesso",      s_known_periph_user),
        ("service_noto_comp",     "service (nodo 5) noto compromesso",              s_known_service),
        ("isolato_noto_sano",     "nodo isolato e noto sano (contenimento da correggere)", s_isolated_clean),
        ("solo_restore_in_corso", "un nodo in restore, resto della rete pulito",    s_restoring_only),
        ("user_e_critical_noti",  "user (0) e critico (7) entrambi noti compromessi", s_user_and_critical),
        ("falso_alert",           "alert su nodo sano, stato ignoto (falso positivo)", s_false_alert),
    ]


def block_controlled(env, policies: dict) -> str:
    """
    Per ogni scenario, tabula l'azione deterministica scelta da ciascuna
    policy (le RL disponibili + heuristic + oracle). Le baseline con stato
    interno vengono resettate prima di ogni scenario, cosi' rispondono al
    solo stato mostrato.
    """
    names = list(policies.keys())
    rows = []
    catalog = _probe_catalog()
    detail_lines = []

    for pname, pdesc, setter in catalog:
        obs, info, snapshot = _make_probe(env, setter)
        row = [pname]
        for n in names:
            pol = policies[n]
            pol.reset()
            act = pol.select_action(obs, info)
            row.append(action_to_str(act))
        rows.append(row)
        detail_lines.append(f"  {pname:<22} {snapshot}")
        detail_lines.append(f"  {'':<22} ({pdesc})")

    headers = ["scenario"] + names
    tbl = _table(headers, rows)

    return "\n".join([
        "Scenari controllati a stato noto",
        "  Ogni riga e' uno stato costruito a mano; ogni colonna e' l'azione",
        "  deterministica scelta da quella policy in quello stato.",
        "",
        tbl,
        "",
        "Stato di ciascuno scenario (per rilettura):",
        "\n".join(detail_lines),
    ])


# D) Confronto con le baseline
def block_baseline_agreement(episodes, policy_name: str) -> tuple[str, dict]:
    """
    Percentuale di passi in cui la policy sceglie la stessa azione della heuristic
    sulla medesima osservazione, più le divergenze più frequenti per categoria.

    L'accordo e' calcolato sull'azione esatta (tipo + bersaglio): due policy
    che eseguono la stessa azione ma su nodi diversi contano come disaccordo.
    """
    total = 0
    agree = 0
    disagree_cat = Counter()   # (categoria_policy, categoria_heuristic)
    for ep in episodes:
        for h in ep["history"]:
            ref = h.get("ref_action")
            if ref is None:
                continue
            total += 1
            if h["action"] == ref:
                agree += 1
            else:
                pcat = _cat_of_step(h["info"])
                # categoria dell'azione heuristic: decodifica dal solo numero
                hcat = _decode_cat(ref)
                disagree_cat[(pcat, hcat)] += 1

    rate = (100.0 * agree / total) if total else 0.0
    top = disagree_cat.most_common(6)
    dis_rows = [[f"{p} vs {h}", c] for (p, h), c in top]
    tbl = (_table(["azione policy vs heuristic", "conteggio"], dis_rows)
           if dis_rows else "(nessun disaccordo)")

    text = "\n".join([
        f"Confronto con la baseline heuristic — policy: {policy_name}",
        f"  Accordo sull'azione esatta: {agree}/{total} passi ({rate:.1f}%).",
        "",
        "  Divergenze piu' frequenti (categoria della policy vs della heuristic):",
        tbl,
    ])
    key = {"agreement_rate": rate, "agree": agree, "total": total}
    return text, key


def _decode_cat(action: int) -> str:
    """Categoria di un'azione a partire dal solo intero (per la ref heuristic)."""
    if action == ACTION_DO_NOTHING:
        return "do_nothing"
    if 0 <= action <= 7:
        return "analyse"
    if 8 <= action <= 15:
        return "isolate"
    if 16 <= action <= 23:
        return "restore"
    if 24 <= action <= 31:
        return "reconnect"
    return "OTHER"


# E) Episodi rappresentativi
def select_representative(episodes, rare_cats: set) -> list:
    """
    Sceglie fino a tre episodi rappresentativi:

    1) episodio difficile con reward piu' alta, privilegiando quelli conclusi
    con una vittoria;
    2) episodio con reward piu' bassa tra quelli non ancora selezionati;
    3) episodio con il maggior numero di azioni poco frequenti per quella policy;
       a parita', quello con piu' azioni invalide.

    Restituisce una lista di (indice, motivo). Gli episodi selezionati sono distinti.
    """

    def is_hard(ep):
        s = ep["summary"]
        return s.get("critical_ever_compromised", False) or s.get("mean_compromised", 0) >= 2.0

    chosen = []
    used = set()

    # 1) episodio difficile con reward piu' alta
    hard_wins = [(i, ep) for i, ep in enumerate(episodes)
                 if is_hard(ep) and ep["summary"].get("is_win", False)]
    pool = hard_wins if hard_wins else [(i, ep) for i, ep in enumerate(episodes) if is_hard(ep)]
    if pool:
        i = max(pool, key=lambda x: x[1]["total_reward"])[0]
        chosen.append((i, "episodio difficile con reward migliore"))
        used.add(i)

    # 2) episodio con reward piu' bassa tra quelli non ancora selezionati
    remaining = [(i, ep) for i, ep in enumerate(episodes) if i not in used]
    if remaining:
        i = min(remaining, key=lambda x: x[1]["total_reward"])[0]
        chosen.append((i, "episodio rimanente con reward peggiore"))
        used.add(i)

    # 3) episodio con piu' azioni poco frequenti
    def rare_score(ep):
        cnt = 0
        for h in ep["history"]:
            if _cat_of_step(h["info"]) in rare_cats:
                cnt += 1
        inv = ep["summary"].get("invalid_action_rate", 0.0)
        return (cnt, inv)
    remaining = [(i, ep) for i, ep in enumerate(episodes) if i not in used]
    if remaining:
        i = max(remaining, key=lambda x: rare_score(x[1]))[0]
        chosen.append((i, "episodio con piu' azioni rare"))
        used.add(i)

    return chosen


def dump_representative(episodes, chosen, policy_name: str, out_path: Path):
    """Scrive il log passo-passo degli episodi scelti, con legenda e intestazioni."""
    lines = [
        f"Episodi rappresentativi — policy: {policy_name}",
        "=" * 70,
        "",
        state_legend(use_color=False),
        "",
    ]
    for idx, reason in chosen:
        ep = episodes[idx]
        s = ep["summary"]
        lines.append("-" * 70)
        lines.append(f"Episodio #{idx} — {reason}")
        lines.append(
            f"  reward={ep['total_reward']:+.2f}  step={s['n_steps']}  "
            f"vittoria={s.get('is_win')}  sconfitta_tot={s.get('is_wipeout')}  "
            f"critico_mai_comp={not s.get('critical_ever_compromised')}  "
            f"invalide={s.get('invalid_action_rate', 0.0)*100:.1f}%"
        )
        lines.append("")
        for h in ep["history"]:
            line = format_step_log(
                step_n=h["info"]["current_step"],
                action=h["action"],
                reward=h["reward"],
                info=h["info"],
                state_snapshot=h["state_snapshot"],
                use_color=False,
            )
            lines.append(line)
        lines.append("")
    out_path.write_text("\n".join(lines))


def block_representative_summary(episodes, chosen, policy_name: str,
                                 out_file: Path) -> str:
    """Sezione di report: quali episodi, perche', e dove leggerne il dettaglio."""
    rows = []
    for idx, reason in chosen:
        ep = episodes[idx]
        s = ep["summary"]
        rows.append([
            idx, reason.split(" (")[0],
            f"{ep['total_reward']:+.2f}", s["n_steps"],
            str(s.get("is_win")), str(s.get("is_wipeout")),
            f"{s.get('invalid_action_rate', 0.0)*100:.1f}%",
        ])
    tbl = _table(
        ["ep", "motivo", "reward", "step", "vittoria", "sconf.tot", "invalide"],
        rows,
    )
    return "\n".join([
        f"Episodi rappresentativi — policy: {policy_name}",
        tbl,
        "",
        f"  Log passo-passo completo dei tre episodi in: {out_file}",
    ])


# Orchestrazione del report

def analyze_one(env, name, policy, heuristic, seed, n_episodes, out_dir):
    """Esegue i blocchi A, B, D, E per una singola policy RL e stampa il report."""
    episodes = run_rollouts(env, policy, seed, n_episodes, ref_policy=heuristic)

    print("#" * 78)
    print(f"# ANALISI POLICY: {name}")
    print("#" * 78)
    print()

    text_a, key_a = block_action_distribution(episodes)
    print("== A ==", "\n")
    print(text_a, "\n")

    text_b, key_b = block_conditional(episodes)
    print("== B ==", "\n")
    print(text_b, "\n")

    text_d, key_d = block_baseline_agreement(episodes, name)
    print("== D ==", "\n")
    print(text_d, "\n")

    # Categorie rare: frequenza inferiore all'1% dei passi, usate dal terzo
    # criterio di selezione degli episodi rappresentativi
    total = key_a["total_steps"]
    rare_cats = {c for c in ACTION_CATS
                 if key_a["category_counts"].get(c, 0) < 0.01 * total}
    chosen = select_representative(episodes, rare_cats)
    rep_file = out_dir / f"representative_episodes_{name}.txt"
    dump_representative(episodes, chosen, name, rep_file)
    print("== E ==", "\n")
    print(block_representative_summary(episodes, chosen, name, rep_file), "\n")

    return {"action_distribution": key_a, "conditional": key_b, "baseline": key_d}


def main():
    ap = argparse.ArgumentParser(
        description="Analisi del comportamento di una policy appresa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ppo-model-path", type=str, default=None,
                    help="path al .zip del modello PPO da analizzare")
    ap.add_argument("--dqn-model-path", type=str, default=None,
                    help="path al .zip del modello DQN da analizzare")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed condiviso della sequenza di episodi (default 42)")
    ap.add_argument("--n-episodes", type=int, default=50,
                    help="numero di episodi per policy (default 50, "
                         "riconciliabile con la tabella di confronto salvata)")
    ap.add_argument("--out-dir", type=str, default="runs/policy_analysis",
                    help="cartella per gli artefatti (default runs/policy_analysis)")
    ap.add_argument("--no-color", action="store_true",
                    help="ininfluente sul report (che e' sempre senza colore); "
                         "accettato per uniformita' con evaluate.py")
    args = ap.parse_args()

    if args.ppo_model_path is None and args.dqn_model_path is None:
        raise SystemExit("Serve almeno uno fra --ppo-model-path e --dqn-model-path.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Un solo env riusato ovunque: reset() lo re-inizializza per episodio, e
    # negli scenari controllati ne sovrascriviamo il runtime a mano
    env = IncidentResponseEnv(seed=args.seed)
    heuristic = HeuristicDefenderPolicy()
    oracle    = OracleDefenderPolicy()

    # Carica le policy RL richieste
    rl = {}
    if args.ppo_model_path is not None:
        rl["ppo"] = load_rl_policy("ppo", args.ppo_model_path)
    if args.dqn_model_path is not None:
        rl["dqn"] = load_rl_policy("dqn", args.dqn_model_path)

    # Intestazione del report: parametri e nota di riconciliazione
    print("=" * 78)
    print("ANALISI DELLA POLICY APPRESA")
    print("=" * 78)
    print(f"seed={args.seed}  n_episodes={args.n_episodes}")
    if args.ppo_model_path:
        print(f"ppo_model={args.ppo_model_path}")
    if args.dqn_model_path:
        print(f"dqn_model={args.dqn_model_path}")
    print()

    key_numbers = {"seed": args.seed, "n_episodes": args.n_episodes, "policies": {}}

    # Blocchi per-policy (A, B, D, E)
    for name, pol in rl.items():
        key_numbers["policies"][name] = analyze_one(
            env, name, pol, heuristic, args.seed, args.n_episodes, out_dir
        )

    # Blocco C condiviso: confronta tutte le policy disponibili sugli stessi
    # scenari controllati. Ordine colonne: RL disponibili, poi heuristic, poi
    # oracle.
    probe_policies = {}
    probe_policies.update(rl)
    probe_policies["heuristic"] = heuristic
    probe_policies["oracle"]    = oracle
    print("#" * 78)
    print("# SCENARI CONTROLLATI (confronto fra tutte le policy)")
    print("#" * 78)
    print()
    print("== C ==", "\n")
    print(block_controlled(env, probe_policies), "\n")

    # Numeri chiave in JSON, per rilettura futura senza rieseguire nulla.
    (out_dir / "key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, default=str)
    )
    print(f"[artefatti scritti in: {out_dir}/]")


if __name__ == "__main__":
    main()
