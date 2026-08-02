"""
Policy iper-difensiva: isola immediatamente ogni nodo non isolato su cui
compare un alert (falsi positivi inclusi), altrimenti DoNothing

Non disinfetta mai (Restore) e non ricollega mai (Reconnect)
"""
from baselines.policy_base import (
    BasePolicy,
    decode_obs,
    isolate,
    ACTION_DO_NOTHING,
)


class PanicIsolationPolicy(BasePolicy):
    """Isola sul primo alert utile; a piu' candidati preferisce il business value maggiore """
    name = "panic_isolation"

    def select_action(self, obs, info=None) -> int:
        nodes = decode_obs(obs)

        # Candidati: alert visibile e nodo non ancora isolato (per evitare azioni invalide)
        candidates = [
            (i, n) for i, n in enumerate(nodes)
            if n["alert"] and not n["isolated"]
        ]

        if not candidates:
            return ACTION_DO_NOTHING

        # Ordinamento di isolamento: business_value decrescente, poi indice crescente
        candidates.sort(key=lambda x: (-x[1]["business_value"], x[0]))
        target = candidates[0][0]
        return isolate(target)