"""
Base condivisa dalle policy. 
"""
import numpy as np


# Costanti dell'action space (33 azioni)
N_NODES = 8

ACTION_ANALYSE_BASE   = 0      # 0..7
ACTION_ISOLATE_BASE   = 8      # 8..15
ACTION_RESTORE_BASE   = 16     # 16..23
ACTION_RECONNECT_BASE = 24     # 24..31
ACTION_DO_NOTHING     = 32

N_ACTIONS = ACTION_DO_NOTHING + 1   # 33


def analyse(i: int) -> int:
    """Codifica intera dell'azione Analyse(i)."""
    return ACTION_ANALYSE_BASE + i


def isolate(i: int) -> int:
    """Codifica intera dell'azione Isolate(i)."""
    return ACTION_ISOLATE_BASE + i


def restore(i: int) -> int:
    """Codifica intera dell'azione Restore(i)."""
    return ACTION_RESTORE_BASE + i


def reconnect(i: int) -> int:
    """Codifica intera dell'azione Reconnect(i)."""
    return ACTION_RECONNECT_BASE + i


def decode_obs(obs) -> list[dict]:
    """
    Decompone l'osservazione (40,) in 8 dizionari per-nodo con i campi originali: 
    - alert/isolated/restoring (bool), 
    - business_value (1/3/6),
    - analysis_state (0/1/2). 

    Le 5 feature per nodo sono: alert, isolated, restoring, business_value (/6), analysis_state (/2).
    """
    nodes = []
    for i in range(N_NODES):
        base = i * 5

        # soglia 0.5 sui binari: robusta a eventuali normalizzazioni o noise
        # futuri
        alert     = bool(float(obs[base + 0]) >= 0.5)
        isolated  = bool(float(obs[base + 1]) >= 0.5)
        restoring = bool(float(obs[base + 2]) >= 0.5)

        bv  = int(round(float(obs[base + 3]) * 6))   # da {1/6,3/6,6/6} a {1,3,6}
        as_ = int(round(float(obs[base + 4]) * 2))   # da {0,0.5,1.0} a {0,1,2}

        nodes.append({
            "alert":          alert,
            "isolated":       isolated,
            "restoring":      restoring,
            "business_value": bv,
            "analysis_state": as_,
        })
    return nodes


class BasePolicy:
    """
    Interfaccia minima per una policy: 
    - select_action(obs, info=None) -> int in [0, 32]; 
    - reset() opzionale per azzerare lo stato a inizio episodio.
    """

    name: str = "base"

    def reset(self) -> None:
        """Chiamata a inizio episodio (default: nessuno stato da azzerare) """
        pass

    def select_action(self, obs, info=None) -> int:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """
    Sceglie un'azione a caso tra le 33, senza guardare l'osservazione.

    Il generatore casuale viene creato una volta sola in __init__ a partire
    dal seed: dato lo stesso seed, produce sempre la stessa sequenza di azioni
    (quindi riproducibile)
    """

    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs, info=None) -> int:
        return int(self._rng.integers(0, N_ACTIONS))