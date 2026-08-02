"""
Ambiente Gymnasium custom IncidentResponseEnv: simulazione di Incident
Response su una rete aziendale a 8 nodi.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.config import EnvConfig
from envs.topology import (
    build_network,
    NodeRuntime,
    NODE_TYPE_USER,
    BUSINESS_VALUE,
)


class IncidentResponseEnv(gym.Env):
    """
    Ambiente Gymnasium per Incident Response su rete a 8 nodi (5 user,
    2 service, 1 critical). 
    Un attaccante stocastico si propaga; il Blue Team lo contiene minimizzando 
    danno da compromissione e costo delle contromisure.

    Observation: Box(40,) = 5 feature x 8 nodi.  Action: Discrete(33)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig = None, seed: int = None):
        super().__init__()

        self.cfg = config if config is not None else EnvConfig()

        # Topologia fissa
        self.network = build_network()

        # Solo gli user node possono essere patient_zero
        self.user_node_ids = [
            s.node_id for s in self.network if s.node_type == NODE_TYPE_USER
        ]

        # Business value normalizzato in [0, 1] per l'osservazione (diviso per 6)
        self.bv_normalized = {
            s.node_id: s.business_value / 6.0 for s in self.network
        }

        # Observation: 8 nodi x 5 feature = 40 valori in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.cfg.n_nodes * 5,),
            dtype=np.float32,
        )

        # 33 azioni discrete:
        #   0..7   = Analyse(i)
        #   8..15  = Isolate(i)
        #   16..23 = Restore(i)
        #   24..31 = Reconnect(i)
        #   32     = DoNothing
        self.action_space = spaces.Discrete(
            self.cfg.n_nodes_actions * 4 + 1
        )

        # Stato episodico (inizializzato da reset)
        self.runtime: list[NodeRuntime] = []
        self.current_step: int = 0
        self.clean_streak: int = 0
        self.patient_zero: int = -1            # -1 = non ancora inizializzato

        self._np_random = np.random.default_rng(seed)

    def reset(self, seed: int = None, options: dict = None):
        """Inizializza un nuovo episodio e restituisce (obs, info)."""
        # Re-seed dall'esterno, richiesto da Gymnasium
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Tutti i nodi sani
        self.runtime = [NodeRuntime() for _ in range(self.cfg.n_nodes)]

        # patient_zero scelto tra i soli user node
        self.patient_zero = int(
            self._np_random.choice(self.user_node_ids)
        )
        self.runtime[self.patient_zero].compromised = True

        self.current_step = 0
        self.clean_streak = 0

        # Alert iniziali con la normale logica osservativa 
        self._update_alerts()

        obs = self._get_obs()
        info = self._get_info(action_was_invalid=False,
                              action_type="reset",
                              action_target=None)

        return obs, info

    def _get_obs(self) -> np.ndarray:
        """
        Costruisce l'osservazione parziale (40,). Per ogni nodo, 5 feature:
        [alert, isolated, restoring, business_value_norm, analysis_state_norm]

        L'agente non osserva 'compromised' ne' il valore di 'restore_timer'
        """
        features = []

        for i, rt in enumerate(self.runtime):
            alert    = 1.0 if rt.alert_active else 0.0
            isolated = 1.0 if rt.isolated else 0.0
            restoring = 1.0 if rt.restore_timer > 0 else 0.0
            bv_norm  = self.bv_normalized[i]

            # analysis_state 0/1/2 normalizzato a 0.0/0.5/1.0
            an_norm = rt.analysis_state * 0.5

            features.extend([alert, isolated, restoring, bv_norm, an_norm])

        return np.array(features, dtype=np.float32)

    def _update_alerts(self):
        """
        Aggiorna alert_active con rumore:
        - Se il nodo è in restore (offline) → nessun alert; 
        - se compromesso → alert con p=0.70; 
        - altrimenti (sano) → falso positivo con p=0.10.
        """
        for rt in self.runtime:
            if rt.restore_timer > 0:
                rt.alert_active = False
            elif rt.compromised:
                rt.alert_active = bool(
                    self._np_random.random() < self.cfg.p_alert_true
                )
            else:
                rt.alert_active = bool(
                    self._np_random.random() < self.cfg.p_alert_false
                )

    def _get_info(self,
                  action_was_invalid: bool,
                  action_type: str,
                  action_target,
                  action_type_original: str = None) -> dict:
        """
        Dizionario di diagnosi restituito da reset() e step()

        - action_type vale "invalid" quando l'azione e' rifiutata
        - action_type_original conserva sempre il tipo decodificato, 
            anche se invalida (utile per l'analisi della policy), 
            ed e' None solo dopo reset().
        """
        compromised_nodes = [
            i for i, rt in enumerate(self.runtime) if rt.compromised
        ]
        isolated_nodes = [
            i for i, rt in enumerate(self.runtime)
            if rt.isolated and rt.restore_timer == 0
        ]
        restoring_nodes = [
            i for i, rt in enumerate(self.runtime) if rt.restore_timer > 0
        ]

        return {
            "compromised_nodes":       compromised_nodes,
            "isolated_nodes":          isolated_nodes,
            "restoring_nodes":         restoring_nodes,
            "n_compromised":           len(compromised_nodes),
            "total_compromised_value": sum(
                self.network[i].business_value for i in compromised_nodes
            ),
            "action_was_invalid":      action_was_invalid,
            "action_type":             action_type,
            "action_type_original":    action_type_original,
            "action_target":           action_target,
            "clean_streak":            self.clean_streak,
            "current_step":            self.current_step,
        }

    def _decode_action(self, action: int) -> tuple[str, int | None]:
        """
        Traduce l'intero dell'agente in (tipo, nodo_target). Non verifica la
        validita' logica (lo fa _is_valid_action() )
        """
        if action == 32:
            return "do_nothing", None
        elif 0 <= action <= 7:
            return "analyse", action
        elif 8 <= action <= 15:
            return "isolate", action - 8
        elif 16 <= action <= 23:
            return "restore", action - 16
        elif 24 <= action <= 31:
            return "reconnect", action - 24
        else:
            raise ValueError(f"Azione fuori range: {action}")

    def _is_valid_action(self, action_type: str, target: int | None) -> bool:
        """
        True se l'azione e' applicabile nello stato corrente del target.
        
        Sono invalide: 
        - Isolate su nodo gia' isolato, 
        - Reconnect su nodo non isolato o in restore, 
        - Restore su nodo gia' in restore. 
        - DoNothing e Analyse sono sempre valide.
        """
        if action_type == "do_nothing":
            return True

        rt = self.runtime[target]

        if action_type == "analyse":
            return True

        elif action_type == "isolate":
            return not rt.isolated

        elif action_type == "restore":
            return rt.restore_timer == 0

        elif action_type == "reconnect":
            return rt.isolated and rt.restore_timer == 0

        return False

    def _apply_analyse(self, i: int):
        """Analyse(i): rivela lo stato reale del nodo (analysis_state 1=sano, 2=compromesso)"""
        rt = self.runtime[i]
        rt.analysis_state = 2 if rt.compromised else 1

    def _apply_isolate(self, i: int):
        """Isolate(i): disconnette il nodo (se infetto, resta compromesso)"""
        self.runtime[i].isolated = True

    def _apply_restore(self, i: int):
        """
        Restore(i): porta offline il nodo, e inizia il processo di riparazione.
        
        restore_timer parte da 3, ma il decremento immediato in step() lo porta
        a 2: il nodo resta offline 2 step effettivi (turno corrente incluso)
        """
        rt = self.runtime[i]
        rt.compromised    = False
        rt.isolated       = True
        rt.restore_timer  = self.cfg.restore_timer_init   # = 3
        rt.alert_active   = False
        rt.analysis_state = 0

    def _apply_reconnect(self, i: int):
        """
        Reconnect(i): riporta online un nodo isolato
        
        Se il nodo era compromesso e isolato, torna compromesso e operativo:
        possibile errore dell'agente.
        """
        self.runtime[i].isolated = False

    def step(self, action: int):
        """
        Esegue un turno completo, nell'ordine:
        1. decodifica 
        2. azione di difesa  
        3. decremento timer restore
        4. propagazione  
        5. alert  
        6. osservazione  
        7. reward
        8. contatori  
        9. terminated/truncated  
        10. info

        Restituisce (obs, reward, terminated, truncated, info)
        """
        # 1. Decodifica azione
        action_type, target = self._decode_action(action)

        # 2. Azione di difesa (solo se valida)
        valid = self._is_valid_action(action_type, target)

        if not valid:
            action_was_invalid = True
        else:
            action_was_invalid = False
            if action_type == "analyse":
                self._apply_analyse(target)
            elif action_type == "isolate":
                self._apply_isolate(target)
            elif action_type == "restore":
                self._apply_restore(target)
            elif action_type == "reconnect":
                self._apply_reconnect(target)
            # do_nothing: nessun effetto

        # 3. Decremento timer restore. Un nodo che torna online qui e' subito
        #    esposto alla propagazione nello stesso turno
        for rt in self.runtime:
            if rt.restore_timer > 0:
                rt.restore_timer -= 1
                if rt.restore_timer == 0:
                    rt.isolated = False
                    rt.analysis_state = 0

        # 4. Propagazione attaccante: ogni nodo compromesso, online e non in
        #    restore tenta di infettare un vicino infettabile con p_spread.
        for i, rt in enumerate(self.runtime):
            if not rt.compromised:
                continue
            if rt.isolated:
                continue
            if rt.restore_timer > 0:
                continue

            infectable = [
                nb for nb in self.network[i].neighbors
                if not self.runtime[nb].compromised
                and not self.runtime[nb].isolated
                and self.runtime[nb].restore_timer == 0
            ]

            if not infectable:
                continue

            chosen = int(self._np_random.choice(infectable))
            if self._np_random.random() < self.cfg.p_spread:
                self.runtime[chosen].compromised = True
                # la conoscenza "noto sano" non e' piu' valida
                if self.runtime[chosen].analysis_state == 1:
                    self.runtime[chosen].analysis_state = 0

        # 5. Alert
        self._update_alerts()

        # 6. Osservazione
        obs = self._get_obs()

        # 7. Reward
        #    Doppio conteggio: un nodo in restore pesa solo in R_t; 
        #    un nodo compromesso e isolato (non in restore) pesa sia in
        #    C_t sia in I_t
        reward = 0.0

        for i, rt in enumerate(self.runtime):
            v = self.network[i].business_value

            if rt.restore_timer > 0:
                reward -= self.cfg.gamma * v       # R_t
            else:
                if rt.compromised:
                    reward -= self.cfg.alpha * v   # C_t
                if rt.isolated:
                    reward -= self.cfg.beta * v    # I_t

        if action_was_invalid:
            reward -= self.cfg.eta

        # 8. Contatori
        #    clean_streak avanza solo se la rete e' davvero
        #    stabilizzata: nessun nodo compromesso e nessuno in restore
        self.current_step += 1

        any_compromised = any(rt.compromised for rt in self.runtime)
        any_restoring   = any(rt.restore_timer > 0 for rt in self.runtime)

        if not any_compromised and not any_restoring:
            self.clean_streak += 1
        else:
            self.clean_streak = 0

        # 9. Terminazione: vittoria/sconfitta (terminated) vs timeout (truncated)
        all_compromised = all(rt.compromised for rt in self.runtime)
        terminated = all_compromised or (self.clean_streak >= self.cfg.clean_streak_target)
        truncated  = self.current_step >= self.cfg.max_steps

        # 10. Info
        info = self._get_info(
            action_was_invalid=action_was_invalid,
            action_type=action_type if valid else "invalid",
            action_target=target,
            action_type_original=action_type,
        )

        return obs, reward, terminated, truncated, info

    def render(self):
        """Non usato in questo progetto (nessuna modalita' di render)."""
        pass