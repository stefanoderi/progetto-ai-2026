"""
Helper per:
- formattazione di azioni/stato, 
- logging step-by-step, 
- sintesi e aggregazione degli episodi. 
Le funzioni di formattazione accettano use_color (codici ANSI, 
attivati automaticamente da supports_color() se stdout e' un tty).
"""
import sys
from typing import Optional

from envs.incident_response_env import IncidentResponseEnv
from envs.topology import (
    NodeRuntime, NodeSpec,
    NODE_TYPE_USER, NODE_TYPE_SERVICE, NODE_TYPE_CRITICAL,
)


# Codici colore ANSI (16 colori base). 
# Il RESET va emesso dopo ogni colore, altrimenti il codice ansi resta attivo sull'output successivo
class _ANSI:
    RESET     = "\033[0m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"

    # foreground
    RED       = "\033[31m"
    GREEN     = "\033[32m"
    YELLOW    = "\033[33m"
    BLUE      = "\033[34m"
    MAGENTA   = "\033[35m"
    CYAN      = "\033[36m"
    GRAY      = "\033[90m"

    # bright
    BRIGHT_RED    = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_GREEN  = "\033[92m"


def supports_color(force_off: bool = False) -> bool:
    """
    True se conviene emettere codici ANSI: mai con force_off (--no-color) o
    quando stdout non e' un terminale (es. redirect su file), cosi' i codici
    non finiscono scritti come testo dentro il log
    """
    if force_off:
        return False
    return sys.stdout.isatty()


def _color(text: str, color_code: str, use_color: bool) -> str:
    """Avvolge text nel codice colore solo se use_color=True."""
    if not use_color:
        return text
    return f"{color_code}{text}{_ANSI.RESET}"


# 1) Formattazione di azioni e stato

def action_to_str(action: int) -> str:
    """Converte un'azione intera in stringa leggibile (es. 10 -> 'Isolate(2)', 32 -> 'DoNothing')"""
    if action == 32:
        return "DoNothing"
    if 0 <= action <= 7:
        return f"Analyse({action})"
    if 8 <= action <= 15:
        return f"Isolate({action - 8})"
    if 16 <= action <= 23:
        return f"Restore({action - 16})"
    if 24 <= action <= 31:
        return f"Reconnect({action - 24})"
    raise ValueError(f"action fuori range [0,32]: {action}")


def _node_type_letter(node_type: str) -> str:
    """Lettera maiuscola del tipo di nodo"""
    return {
        NODE_TYPE_USER:     "U",
        NODE_TYPE_SERVICE:  "S",
        NODE_TYPE_CRITICAL: "C",
    }[node_type]


def format_node_compact(spec: NodeSpec, runtime: NodeRuntime,
                        use_color: bool = False) -> str:
    """
    Formatta un nodo in modo compatto: 'id:Tipo[c i R a ?]', con 5 caratteri
    fissi dentro le quadre (compromised, isolated, Restore, alert, analysis:
    ?=sconosciuto / h=noto sano / X=noto compromesso)
    """
    c1 = _color("c", _ANSI.RED, use_color)              if runtime.compromised   else "."
    # In restore il nodo e' anche isolated, ma lo mostriamo come 'R', non 'i':
    # per questo controlliamo restore_timer prima
    in_restore = runtime.restore_timer > 0
    c2 = "."
    if runtime.isolated and not in_restore:
        c2 = _color("i", _ANSI.BLUE, use_color)
    c3 = _color("R", _ANSI.YELLOW, use_color)           if in_restore           else "."
    c4 = _color("a", _ANSI.MAGENTA, use_color)          if runtime.alert_active  else "."

    if   runtime.analysis_state == 0:
        c5 = "?"
    elif runtime.analysis_state == 1:
        c5 = _color("h", _ANSI.GREEN, use_color)
    else:  # 2
        c5 = _color("X", _ANSI.BRIGHT_RED, use_color)

    type_letter = _node_type_letter(spec.node_type)
    if spec.node_type == NODE_TYPE_CRITICAL:
        type_letter = _color(type_letter, _ANSI.CYAN, use_color)

    return f"{spec.node_id}:{type_letter}[{c1}{c2}{c3}{c4}{c5}]"


def format_state(env: IncidentResponseEnv, use_color: bool = False) -> str:
    """Snapshot compatto di tutti i nodi della rete, separati da spazio."""
    parts = [
        format_node_compact(spec, runtime, use_color)
        for spec, runtime in zip(env.network, env.runtime)
    ]
    return " ".join(parts)


def state_legend(use_color: bool = False) -> str:
    """Legenda dei simboli di format_node_compact, da stampare una volta a inizio log."""
    c = lambda txt, col: _color(txt, col, use_color)
    lines = [
        "Legenda dello stato dei nodi (formato: id:Tipo[c i R a ?]):",
        f"  Tipo:   U=User  S=Service  {c('C', _ANSI.CYAN)}=Critical",
        f"  c = {c('compromesso', _ANSI.RED)}",
        f"  i = isolato (non in restore)",
        f"  R = {c('in restore', _ANSI.YELLOW)}",
        f"  a = {c('alert attivo', _ANSI.MAGENTA)}",
        f"  ? = analysis sconosciuto,  h = noto sano,  "
        f"{c('X', _ANSI.BRIGHT_RED)} = noto compromesso",
    ]
    return "\n".join(lines)


# 2) Logging step-by-step

def format_step_log(step_n: int, action: int, reward: float,
                    info: dict, state_snapshot: str,
                    use_color: bool = False) -> str:
    """
    Riga di log per uno step, larghezza ~costante per leggerla in colonna.
    state_snapshot va passato gia' calcolato al momento dello step (non l'env):
    l'env potrebbe essere gia' avanzato e produrrebbe snapshot tutti uguali.
    """
    inv = "1" if info.get("action_was_invalid", False) else "0"
    inv_colored = (_color(inv, _ANSI.BRIGHT_RED, use_color)
                   if inv == "1" else inv)

    # Reward: rosso se molto negativa, gialla se moderata, verde se >= 0
    rew_str = f"{reward:+.2f}"
    if use_color:
        if reward <= -3.0:
            rew_str = _color(rew_str, _ANSI.RED, use_color)
        elif reward < 0:
            rew_str = _color(rew_str, _ANSI.YELLOW, use_color)
        else:
            rew_str = _color(rew_str, _ANSI.GREEN, use_color)

    return (
        f"[t={step_n:02d}] "
        f"act={action_to_str(action):<14} "
        f"inv={inv_colored} "
        f"rew={rew_str:>6} "
        f"cs={info['clean_streak']} "
        f"nC={info['n_compromised']} "
        f"vC={info['total_compromised_value']:<2} "
        f"| {state_snapshot}"
    )


# 3) Sintesi episodica e aggregazione

def summarize_episode(history: list[dict], total_reward: float,
                      terminated: bool, truncated: bool,
                      env: IncidentResponseEnv) -> dict:
    """
    Statistiche di sintesi di un episodio concluso, a partire dalla history dei
    dict info (uno per step). I campi restituiti sono quelli del dict finale.
    """
    n_steps = len(history)
    if n_steps == 0:
        # Episodio senza step: valori neutri
        return {
            "n_steps":                  0,
            "total_reward":             0.0,
            "terminated":               terminated,
            "truncated":                truncated,
            "is_win":                   False,
            "is_wipeout":               False,
            "mean_compromised":         0.0,
            "mean_compromised_value":   0.0,
            "invalid_action_rate":      0.0,
            "critical_ever_compromised": False,
            "action_counts":            {},
            "target_type_counts":       {},
        }

    mean_compromised = sum(h["n_compromised"] for h in history) / n_steps
    mean_compromised_value = (
        sum(h["total_compromised_value"] for h in history) / n_steps
    )
    invalid_action_rate = (
        sum(1 for h in history if h.get("action_was_invalid", False)) / n_steps
    )

    # Critical = nodo 7 (topologia fissa): controlla se e' mai stato compromesso
    critical_id = 7
    critical_ever_compromised = any(
        critical_id in h["compromised_nodes"] for h in history
    )

    # Distinzione vittoria/sconfitta dentro terminated: lo stesso flag copre
    # rete bonificata (clean_streak >= target) e rete distrutta (tutti
    # compromessi). 
    # Nei casi rari in cui terminated e truncated coesistono
    # (caduta esatta al 40esimo step) non forziamo una classificazione e
    # lasciamo entrambi i flag a False
    is_win = False
    is_wipeout = False
    if terminated:
        last_info = history[-1]
        target = env.cfg.clean_streak_target
        if last_info.get("clean_streak", 0) >= target:
            is_win = True
        elif last_info.get("n_compromised", 0) >= env.cfg.n_nodes:
            is_wipeout = True

    # Distribuzione delle azioni per tipo
    action_counts = {}
    for h in history:
        atype = h.get("action_type", "unknown")
        action_counts[atype] = action_counts.get(atype, 0) + 1

    # Distribuzione dei target per tipo di nodo (solo azioni valide con
    # target esplicito: analyse/isolate/restore/reconnect)
    target_type_counts = {}
    valid_action_types = {"analyse", "isolate", "restore", "reconnect"}
    for h in history:
        if h.get("action_type") in valid_action_types and h.get("action_target") is not None:
            target_id = h["action_target"]
            ntype = env.network[target_id].node_type
            target_type_counts[ntype] = target_type_counts.get(ntype, 0) + 1

    return {
        "n_steps":                  n_steps,
        "total_reward":             total_reward,
        "terminated":               terminated,
        "truncated":                truncated,
        "is_win":                   is_win,
        "is_wipeout":               is_wipeout,
        "mean_compromised":         mean_compromised,
        "mean_compromised_value":   mean_compromised_value,
        "invalid_action_rate":      invalid_action_rate,
        "critical_ever_compromised": critical_ever_compromised,
        "action_counts":            action_counts,
        "target_type_counts":       target_type_counts,
    }


def format_episode_summary(summary: dict, episode_id: Optional[int] = None,
                           use_color: bool = False) -> str:
    """Pretty-printer multi-line di una sintesi di episodio."""
    title = "Episodio"
    if episode_id is not None:
        title += f" #{episode_id}"

    # Priorita' terminated > truncated: se coesistono (raro, caduta al limite
    # di step) mostriamo terminated, che e' la causa reale
    if summary["terminated"]:
        end_str = _color("terminated", _ANSI.GREEN, use_color)
    elif summary["truncated"]:
        end_str = _color("truncated", _ANSI.YELLOW, use_color)
    else:
        end_str = "?"

    crit_str = (_color("SI", _ANSI.BRIGHT_RED, use_color)
                if summary["critical_ever_compromised"]
                else _color("no", _ANSI.GREEN, use_color))

    action_counts = summary["action_counts"]
    actions_line = ", ".join(f"{k}={v}" for k, v in sorted(action_counts.items()))

    target_counts = summary["target_type_counts"]
    targets_line = ", ".join(f"{k}={v}" for k, v in sorted(target_counts.items()))

    lines = [
        f"=== {title} ===",
        f"  step:                {summary['n_steps']}",
        f"  reward totale:       {summary['total_reward']:+.2f}",
        f"  fine episodio:       {end_str}",
        f"  media compromessi:   {summary['mean_compromised']:.2f}",
        f"  media valore comp.:  {summary['mean_compromised_value']:.2f}",
        f"  azioni invalide:     {summary['invalid_action_rate']*100:.1f}%",
        f"  critical mai comp.?: {crit_str}",
        f"  azioni:              {actions_line}",
        f"  target per tipo:     {targets_line if targets_line else '(nessuno)'}",
    ]
    return "\n".join(lines)


def aggregate_episode_summaries(summaries: list[dict]) -> dict:
    """Aggrega N sintesi episodiche in statistiche cross-episode (campi nel dict finale)."""
    n = len(summaries)
    if n == 0:
        return {
            "n_episodes":                 0,
            "mean_total_reward":          0.0,
            "std_total_reward":           0.0,
            "terminated_rate":            0.0,
            "truncated_rate":             0.0,
            "win_rate":                   0.0,
            "wipeout_rate":               0.0,
            "mean_n_steps":               0.0,
            "mean_invalid_action_rate":   0.0,
            "critical_compromised_rate":  0.0,
            "action_counts_total":        {},
        }

    rewards = [s["total_reward"] for s in summaries]
    mean_r = sum(rewards) / n
    var_r  = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r  = var_r ** 0.5

    terminated_rate = sum(1 for s in summaries if s["terminated"]) / n
    truncated_rate  = sum(1 for s in summaries if s["truncated"]) / n

    # win_rate e wipeout_rate sono sotto-categorie di terminated
    win_rate     = sum(1 for s in summaries if s.get("is_win", False))     / n
    wipeout_rate = sum(1 for s in summaries if s.get("is_wipeout", False)) / n

    mean_n_steps = sum(s["n_steps"] for s in summaries) / n
    mean_inv_rate = sum(s["invalid_action_rate"] for s in summaries) / n

    crit_rate = sum(1 for s in summaries
                    if s["critical_ever_compromised"]) / n

    action_counts_total = {}
    for s in summaries:
        for k, v in s["action_counts"].items():
            action_counts_total[k] = action_counts_total.get(k, 0) + v

    return {
        "n_episodes":                n,
        "mean_total_reward":         mean_r,
        "std_total_reward":          std_r,
        "terminated_rate":           terminated_rate,
        "truncated_rate":            truncated_rate,
        "win_rate":                  win_rate,
        "wipeout_rate":              wipeout_rate,
        "mean_n_steps":              mean_n_steps,
        "mean_invalid_action_rate":  mean_inv_rate,
        "critical_compromised_rate": crit_rate,
        "action_counts_total":       action_counts_total,
    }


def format_aggregate_summary(agg: dict, use_color: bool = False) -> str:
    """Pretty-printer della sintesi aggregata su N episodi."""
    actions_line = ", ".join(f"{k}={v}"
                             for k, v in sorted(agg["action_counts_total"].items()))

    # Verde per le vittorie, rosso per le sconfitte totali
    win_str = f"{agg.get('win_rate', 0.0)*100:.1f}%"
    wipe_str = f"{agg.get('wipeout_rate', 0.0)*100:.1f}%"
    if use_color:
        win_str  = _color(win_str,  _ANSI.GREEN,      use_color)
        wipe_str = _color(wipe_str, _ANSI.BRIGHT_RED, use_color)

    lines = [
        "=" * 60,
        f"SINTESI AGGREGATA SU {agg['n_episodes']} EPISODI",
        "=" * 60,
        f"  reward medio:               {agg['mean_total_reward']:+.2f} "
        f"(σ={agg['std_total_reward']:.2f})",
        f"  step medi per episodio:     {agg['mean_n_steps']:.1f}",
        f"  tasso terminated:           {agg['terminated_rate']*100:.1f}%",
        f"    di cui vittorie:          {win_str}",
        f"    di cui sconfitte totali:  {wipe_str}",
        f"  tasso truncated:            {agg['truncated_rate']*100:.1f}%",
        f"  tasso azioni invalide:      {agg['mean_invalid_action_rate']*100:.1f}%",
        f"  episodi con critical comp.: "
        f"{agg['critical_compromised_rate']*100:.1f}%",
        f"  totale azioni per tipo:     {actions_line}",
        "=" * 60,
    ]
    return "\n".join(lines)