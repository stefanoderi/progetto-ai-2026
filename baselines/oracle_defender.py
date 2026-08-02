"""
Policy oracolo: legge lo stato reale da info["compromised_nodes"], rompendo l'osservabilita' parziale. 
Rappresenta il tetto superiore possibile.
"""
from baselines.policy_base import (
    BasePolicy,
    decode_obs,
    restore,
    reconnect,
    ACTION_DO_NOTHING,
    N_NODES,
)


class OracleDefenderPolicy(BasePolicy):
    """
    Fa solo Reconnect sui sani isolati, Restore su ogni compromesso, altrimenti DoNothing. 
    """

    name = "oracle"

    def select_action(self, obs, info=None) -> int:
        # Senza info al primo step (es. test o se non passato al 1o step): fallback a DoNothing.
        if info is None:
            return ACTION_DO_NOTHING

        compromised_set = set(info.get("compromised_nodes", []))
        nodes = decode_obs(obs)

        # Stesso ordinamento delle altre baseline
        order = sorted(
            range(N_NODES),
            key=lambda i: (-nodes[i]["business_value"], i),
        )

        # Priorita' 1: Reconnect sui nodi isolati e sani
        for i in order:
            n = nodes[i]
            if (n["isolated"]
                    and not n["restoring"]
                    and i not in compromised_set):
                return reconnect(i)

        # Priorita' 2: Restore su ogni compromesso non gia' in restore
        for i in order:
            n = nodes[i]
            if i in compromised_set and not n["restoring"]:
                return restore(i)

        # Priorita' 3: nessuna azione utile
        return ACTION_DO_NOTHING