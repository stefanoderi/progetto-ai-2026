"""
Baseline euristica
"""
from baselines.policy_base import (
    BasePolicy,
    decode_obs,
    analyse,
    isolate,
    restore,
    reconnect,
    ACTION_DO_NOTHING,
    N_NODES,
)


# Turni di "noto compromesso" prima di passare da Isolate a Restore
DEFAULT_RESTORE_AFTER = 2

# Business value del nodo critical (restore immediato)
CRITICAL_BV = 6


class HeuristicDefenderPolicy(BasePolicy):
    """
    Strategia a priorita' decrescente: 
    (1) Reconnect su noto sano isolato, 
    (2) Restore sul critical subito e sugli altri dopo una soglia di persistenza, 
    (3) Isolate sui noti compromessi, 
    (4) Analyse sugli alert sconosciuti, 
    (5) DoNothing. 
    Dentro ogni livello scandisce per business value decrescente, poi indice crescente. 

    I filtri di ogni livello fanno si' che non emetta mai azioni invalide.
    """

    name = "heuristic"

    def __init__(self, restore_after_n_steps: int = DEFAULT_RESTORE_AFTER):
        super().__init__()
        self.restore_after_n_steps = restore_after_n_steps
        # Turni consecutivi in cui il nodo risulta "noto compromesso"
        self._known_comp_streak = [0] * N_NODES

    def reset(self) -> None:
        """Azzera lo stato interno a inizio episodio."""
        self._known_comp_streak = [0] * N_NODES

    def select_action(self, obs, info=None) -> int:
        nodes = decode_obs(obs)

        # Aggiorna i contatori. Dopo un Restore l'env riporta analysis_state
        # a 0, quindi il ramo "else" azzera correttamente la streak.
        for i, n in enumerate(nodes):
            if n["analysis_state"] == 2:        # noto compromesso
                self._known_comp_streak[i] += 1
            else:                                # sconosciuto o noto sano
                self._known_comp_streak[i] = 0

        # Ordine di scansione: business value decrescente, indice crescente
        order = sorted(
            range(N_NODES),
            key=lambda i: (-nodes[i]["business_value"], i),
        )

        # Priorita' 1: Reconnect su noto sano isolato
        for i in order:
            n = nodes[i]
            if (n["isolated"]
                    and not n["restoring"]
                    and n["analysis_state"] == 1):
                return reconnect(i)

        # Priorita' 2a: Restore immediato sul critical
        for i in order:
            n = nodes[i]
            if (n["analysis_state"] == 2
                    and not n["restoring"]
                    and n["business_value"] == CRITICAL_BV):
                return restore(i)

        # Priorita' 2b: Restore sugli altri dopo la soglia di persistenza
        for i in order:
            n = nodes[i]
            if (n["analysis_state"] == 2
                    and not n["restoring"]
                    and self._known_comp_streak[i] >= self.restore_after_n_steps):
                return restore(i)

        # Priorita' 3: Isolate su noto compromesso non isolato
        for i in order:
            n = nodes[i]
            if n["analysis_state"] == 2 and not n["isolated"]:
                return isolate(i)

        # Priorita' 4: Analyse su alert sconosciuto. 
        # Il filtro not restoring e' ridondante (in restore l'env forza alert=False) 
        # ma aumenta robustezza
        for i in order:
            n = nodes[i]
            if (n["alert"]
                    and n["analysis_state"] == 0
                    and not n["restoring"]):
                return analyse(i)

        # Priorita' 5: nessuna azione giustificata
        return ACTION_DO_NOTHING