"""
Wrapper che espone un modello Stable-Baselines3 (DQN, PPO, ...) attraverso
l'interfaccia BasePolicy, cosi' che evaluate.py tratti un agente RL come una
qualsiasi baseline: stessi episodi, stessi seed, stesse metriche. 
"""
from baselines.policy_base import BasePolicy


class SB3PolicyWrapper(BasePolicy):
    """
    Adapter da modello SB3 a BasePolicy:
    - `model` e' un'istanza gia' caricata con metodo predict(obs, deterministic=...)
    - `name` e' l'etichetta usata in log e tabelle
    - 'deterministic=True' (default)
    - `info` ignorato dall'agente
    """

    def __init__(self, model, deterministic: bool = True, name: str = "sb3"):
        super().__init__()
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def select_action(self, obs, info=None) -> int:
        # cast a int: SB3 restituisce a volte np.int64 0-d, a volte shape-(1,);
        # le altre policy danno gia' int Python, qui uniformiamo. (state
        # scartato: None per policy non-ricorrenti)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)