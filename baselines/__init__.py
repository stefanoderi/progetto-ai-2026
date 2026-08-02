"""
Pacchetto delle policy difensive. Espone l'interfaccia BasePolicy, le baseline
(Random, DoNothing, PanicIsolation, HeuristicDefender, OracleDefender), il
wrapper SB3PolicyWrapper e gli helper di decodifica/costruzione delle azioni.
Tutte le policy sono interscambiabili nell'orchestratore evaluate.py.
"""
from baselines.policy_base import (
    BasePolicy,
    RandomPolicy,
    decode_obs,
    analyse,
    isolate,
    restore,
    reconnect,
    ACTION_ANALYSE_BASE,
    ACTION_ISOLATE_BASE,
    ACTION_RESTORE_BASE,
    ACTION_RECONNECT_BASE,
    ACTION_DO_NOTHING,
    N_NODES,
    N_ACTIONS,
)
from baselines.do_nothing import DoNothingPolicy
from baselines.panic_isolation import PanicIsolationPolicy
from baselines.heuristic_defender import HeuristicDefenderPolicy
from baselines.oracle_defender import OracleDefenderPolicy
from baselines.sb3_wrapper import SB3PolicyWrapper


__all__ = [
    # interfaccia
    "BasePolicy",
    # policy
    "RandomPolicy",
    "DoNothingPolicy",
    "PanicIsolationPolicy",
    "HeuristicDefenderPolicy",
    "OracleDefenderPolicy",
    "SB3PolicyWrapper",
    # decodifica e helper
    "decode_obs",
    "analyse",
    "isolate",
    "restore",
    "reconnect",
    # costanti
    "ACTION_ANALYSE_BASE",
    "ACTION_ISOLATE_BASE",
    "ACTION_RESTORE_BASE",
    "ACTION_RECONNECT_BASE",
    "ACTION_DO_NOTHING",
    "N_NODES",
    "N_ACTIONS",
]