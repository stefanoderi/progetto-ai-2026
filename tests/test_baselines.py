"""
Suite di test per le policy non-RL.

Copre:
    - DoNothingPolicy: invariante banale (sempre 32)
    - PanicIsolationPolicy: reazione agli alert, tie-breaking per BV,
      fallback su DoNothing
    - HeuristicDefenderPolicy: ordine di priorità (Reconnect → Restore
      critical → Restore standard → Isolate → Analyse → DoNothing),
      gestione dello streak interno, reset
    - OracleDefenderPolicy: lettura da info, priorità BV, gestione
      casi limite (isolato sano, isolato compromesso)
    - decode_obs: denormalizzazione corretta dei valori boundary
    - tasso azioni invalide pari a 0% per costruzione su tutte le
      baseline deterministiche

Ogni test costruisce uno stato controllato chiamando
env.reset() seguito da force_state(), invoca la policy, e verifica
l'azione emessa. Nessun test dipende da N rollout statistici.
"""
import numpy as np
import pytest

from envs.incident_response_env import IncidentResponseEnv
from baselines import (
    DoNothingPolicy,
    PanicIsolationPolicy,
    HeuristicDefenderPolicy,
    OracleDefenderPolicy,
    RandomPolicy,
    decode_obs,
    analyse,
    isolate,
    restore,
    reconnect,
    ACTION_DO_NOTHING,
    N_ACTIONS,
)


# --- Helper functions ---

def force_state(env, node_id, **kwargs):
    """Forza i campi di NodeRuntime di un nodo specifico."""
    rt = env.runtime[node_id]
    for k, v in kwargs.items():
        setattr(rt, k, v)


def force_all_clean(env):
    """Porta tutti i nodi a stato sano, online, fuori restore."""
    for rt in env.runtime:
        rt.compromised    = False
        rt.isolated       = False
        rt.restore_timer  = 0
        rt.alert_active   = False
        rt.analysis_state = 0


# --- Fixture locale ---
# Nota: 'env' è già definita in conftest.py (scope di funzione).
# Qui aggiungiamo una fixture che restituisce un ambiente con stato
# completamente sotto controllo del test (force_all_clean dopo reset).

@pytest.fixture
def clean_env():
    """Ambiente con reset() chiamato e tutti i nodi forzati a sani."""
    e = IncidentResponseEnv(seed=0)
    e.reset(seed=0)
    force_all_clean(e)
    return e


# --- DoNothingPolicy ---

class TestDoNothingPolicy:
    """L'unica invariante: emette sempre 32, qualunque obs."""

    def test_always_returns_do_nothing(self, env):
        policy = DoNothingPolicy()
        obs = env._get_obs()
        for _ in range(50):
            assert policy.select_action(obs) == ACTION_DO_NOTHING

    def test_works_with_none_info(self, env):
        """Coerenza interfaccia: info è opzionale."""
        policy = DoNothingPolicy()
        obs = env._get_obs()
        assert policy.select_action(obs, info=None) == ACTION_DO_NOTHING

    def test_name(self):
        assert DoNothingPolicy().name == "do_nothing"


# --- PanicIsolationPolicy ---

class TestPanicIsolationPolicy:
    """
    Regola: se almeno un nodo ha alert E non è isolato → emetti isolate
    su quello con BV più alto (tie-breaker: indice più basso).
    Altrimenti → DoNothing.
    """

    def test_no_alerts_returns_do_nothing(self, clean_env):
        policy = PanicIsolationPolicy()
        obs = clean_env._get_obs()
        assert policy.select_action(obs) == ACTION_DO_NOTHING

    def test_single_alert_on_user_node_emits_isolate(self, clean_env):
        force_state(clean_env, 2, alert_active=True)
        obs = clean_env._get_obs()
        policy = PanicIsolationPolicy()
        assert policy.select_action(obs) == isolate(2)

    def test_alert_on_already_isolated_node_is_ignored(self, clean_env):
        """
        Costruzione critica: l'isolato con alert NON deve essere ri-isolato
        (sarebbe azione invalida). La policy deve emettere DoNothing se
        è l'unico candidato.
        """
        # Nota: un nodo isolato e non in restore può ancora generare alert
        # (logica in _update_alerts: alert solo se restore_timer==0).
        # Forziamo manualmente l'alert per testare il filtro.
        force_state(clean_env, 3, isolated=True, alert_active=True)
        obs = clean_env._get_obs()
        policy = PanicIsolationPolicy()
        assert policy.select_action(obs) == ACTION_DO_NOTHING

    def test_tie_break_prefers_higher_business_value(self, clean_env):
        """
        Due alert: uno su user (BV=1, nodo 0) e uno su critical (BV=6, nodo 7).
        Deve isolare il critical.
        """
        force_state(clean_env, 0, alert_active=True)
        force_state(clean_env, 7, alert_active=True)
        obs = clean_env._get_obs()
        policy = PanicIsolationPolicy()
        assert policy.select_action(obs) == isolate(7)

    def test_tie_break_same_bv_prefers_lower_index(self, clean_env):
        """Due alert su user (BV=1): nodi 2 e 4. Deve preferire il 2."""
        force_state(clean_env, 2, alert_active=True)
        force_state(clean_env, 4, alert_active=True)
        obs = clean_env._get_obs()
        policy = PanicIsolationPolicy()
        assert policy.select_action(obs) == isolate(2)

    def test_never_emits_invalid_action_in_rollout(self):
        """Su 100 step di rollout reale, mai un'azione invalida."""
        env = IncidentResponseEnv(seed=42)
        env.reset(seed=42)
        policy = PanicIsolationPolicy()

        obs = env._get_obs()
        info = None
        for _ in range(100):
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            assert not info["action_was_invalid"], (
                f"Azione invalida emessa: action={action}, "
                f"info={info}"
            )
            if terminated or truncated:
                obs, info = env.reset()


# --- HeuristicDefenderPolicy ---

class TestHeuristicDefenderPolicy:
    """
    Verifica l'ordine di priorità:
        1) Reconnect su isolato-noto-sano fuori restore
        2a) Restore immediato sul critical noto compromesso
        2b) Restore sugli altri noti compromessi dopo soglia
        3) Isolate su noto compromesso non isolato
        4) Analyse su alert visibile e analysis sconosciuto
        5) DoNothing
    """

    def test_no_action_needed_returns_do_nothing(self, clean_env):
        policy = HeuristicDefenderPolicy()
        obs = clean_env._get_obs()
        assert policy.select_action(obs) == ACTION_DO_NOTHING

    def test_priority_1_reconnect_isolated_known_clean(self, clean_env):
        """Nodo 2: isolato, fuori restore, noto sano → reconnect."""
        force_state(clean_env, 2, isolated=True, analysis_state=1)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()
        assert policy.select_action(obs) == reconnect(2)

    def test_priority_2a_restore_critical_immediate(self, clean_env):
        """Critical (nodo 7) noto compromesso → restore immediato."""
        force_state(clean_env, 7, compromised=True, analysis_state=2)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()
        action = policy.select_action(obs)
        # NB: l'aggiornamento dello streak avviene PRIMA dei test di priorità
        # quindi qui lo streak diventa 1 ma per il critical c'è priorità immediata
        assert action == restore(7)

    def test_priority_2b_restore_non_critical_after_threshold(self, clean_env):
        """
        Service (nodo 5) noto compromesso. Default threshold=2 → restore
        scatta solo dopo il 2° call consecutivo con analysis_state=2.
        """
        force_state(clean_env, 5, compromised=True, analysis_state=2)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()

        # Primo call: streak diventa 1, ancora < 2 → niente restore
        # (priorità 3 dovrebbe scattare → isolate)
        action1 = policy.select_action(obs)
        # Forziamo isolato per simulare lo step seguente
        force_state(clean_env, 5, isolated=True)
        obs2 = clean_env._get_obs()
        # Secondo call: streak diventa 2, ora restore
        action2 = policy.select_action(obs2)
        assert action2 == restore(5)

    def test_priority_3_isolate_known_compromised(self, clean_env):
        """User compromesso noto (analysis_state=2), non isolato → isolate."""
        force_state(clean_env, 3, compromised=True, analysis_state=2)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()
        # Streak = 1 al primo call, sotto soglia 2 → no restore (BV=1 non critical)
        # → priorità 3 scatta
        assert policy.select_action(obs) == isolate(3)

    def test_priority_4_analyse_on_alert_unknown(self, clean_env):
        """Alert visibile su nodo con analysis_state=0 → analyse."""
        force_state(clean_env, 1, alert_active=True)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()
        assert policy.select_action(obs) == analyse(1)

    def test_priority_order_critical_beats_user(self, clean_env):
        """
        User compromesso (nodo 0, BV=1) + critical compromesso (nodo 7, BV=6),
        entrambi noti, entrambi non isolati. Si aspetta restore(7) immediato
        per la priorità sul critical (P2a), che scavalca isolate(0) (P3).
        """
        force_state(clean_env, 0, compromised=True, analysis_state=2)
        force_state(clean_env, 7, compromised=True, analysis_state=2)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()
        assert policy.select_action(obs) == restore(7)

    def test_streak_resets_after_restore(self, clean_env):
        """
        Dopo Restore, l'env imposta analysis_state=0. La policy deve
        azzerare lo streak interno per quel nodo (è la condizione
        else: _known_comp_streak[i] = 0 quando analysis_state != 2).
        """
        force_state(clean_env, 5, compromised=True, analysis_state=2)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()

        # Primo call: streak diventa 1
        policy.select_action(obs)
        assert policy._known_comp_streak[5] == 1

        # Simuliamo lo stato post-Restore (analysis_state torna a 0)
        force_state(clean_env, 5, compromised=False, analysis_state=0,
                    isolated=True, restore_timer=2)
        obs2 = clean_env._get_obs()
        policy.select_action(obs2)
        assert policy._known_comp_streak[5] == 0

    def test_reset_clears_internal_state(self, clean_env):
        """policy.reset() deve azzerare lo streak interno."""
        force_state(clean_env, 5, compromised=True, analysis_state=2)
        obs = clean_env._get_obs()
        policy = HeuristicDefenderPolicy()
        policy.select_action(obs)
        assert policy._known_comp_streak[5] == 1

        policy.reset()
        assert all(s == 0 for s in policy._known_comp_streak)

    def test_never_emits_invalid_action_in_rollout(self):
        """Su 200 step di rollout, la heuristic non emette mai invalide."""
        env = IncidentResponseEnv(seed=7)
        env.reset(seed=7)
        policy = HeuristicDefenderPolicy()

        obs = env._get_obs()
        info = None
        for _ in range(200):
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            assert not info["action_was_invalid"], (
                f"Azione invalida emessa: action={action}, "
                f"info={info}"
            )
            if terminated or truncated:
                obs, info = env.reset()
                policy.reset()


# --- decode_obs — boundary cases ---

class TestDecodeObs:
    """
    Verifica che decode_obs sia l'inverso esatto di _get_obs():
    business_value torna a 1/3/6, analysis_state torna a 0/1/2, le
    binarie a True/False.
    """

    def test_decode_returns_8_dicts(self, env):
        obs = env._get_obs()
        nodes = decode_obs(obs)
        assert len(nodes) == 8

    def test_business_values_recovered(self, env):
        """User=1, Service=3, Critical=6, indipendentemente dallo stato."""
        obs = env._get_obs()
        nodes = decode_obs(obs)
        expected = [1, 1, 1, 1, 1, 3, 3, 6]
        for i, exp in enumerate(expected):
            assert nodes[i]["business_value"] == exp, (
                f"nodo {i}: atteso BV={exp}, ottenuto "
                f"{nodes[i]['business_value']}"
            )

    def test_analysis_state_all_three_levels(self, clean_env):
        """analysis_state 0/1/2 deve essere recuperato come int."""
        force_state(clean_env, 0, analysis_state=0)
        force_state(clean_env, 1, analysis_state=1)
        force_state(clean_env, 2, analysis_state=2)
        obs = clean_env._get_obs()
        nodes = decode_obs(obs)
        assert nodes[0]["analysis_state"] == 0
        assert nodes[1]["analysis_state"] == 1
        assert nodes[2]["analysis_state"] == 2

    def test_binary_fields_are_bool(self, clean_env):
        """alert, isolated, restoring devono essere bool nativi."""
        force_state(clean_env, 4, alert_active=True, isolated=True,
                    restore_timer=2)
        obs = clean_env._get_obs()
        nodes = decode_obs(obs)
        assert isinstance(nodes[4]["alert"], bool)
        assert isinstance(nodes[4]["isolated"], bool)
        assert isinstance(nodes[4]["restoring"], bool)
        assert nodes[4]["alert"] is True
        assert nodes[4]["isolated"] is True
        assert nodes[4]["restoring"] is True

    def test_all_false_when_clean(self, clean_env):
        obs = clean_env._get_obs()
        nodes = decode_obs(obs)
        for i in range(8):
            assert nodes[i]["alert"] is False
            assert nodes[i]["isolated"] is False
            assert nodes[i]["restoring"] is False


# --- RandomPolicy ---

class TestRandomPolicy:
    """Sampling uniforme, riproducibile dato il seed."""

    def test_returns_valid_action_range(self, env):
        policy = RandomPolicy(seed=0)
        obs = env._get_obs()
        for _ in range(200):
            a = policy.select_action(obs)
            assert 0 <= a < N_ACTIONS

    def test_deterministic_given_seed(self, env):
        """Due policy con stesso seed → stessa sequenza."""
        p1 = RandomPolicy(seed=42)
        p2 = RandomPolicy(seed=42)
        obs = env._get_obs()
        seq1 = [p1.select_action(obs) for _ in range(50)]
        seq2 = [p2.select_action(obs) for _ in range(50)]
        assert seq1 == seq2

    def test_different_seeds_give_different_sequences(self, env):
        p1 = RandomPolicy(seed=0)
        p2 = RandomPolicy(seed=1)
        obs = env._get_obs()
        seq1 = [p1.select_action(obs) for _ in range(50)]
        seq2 = [p2.select_action(obs) for _ in range(50)]
        assert seq1 != seq2


# --- OracleDefenderPolicy ---

class TestOracleDefenderPolicy:
    """
    Verifica che Oracle:
      - usi info["compromised_nodes"] come ground truth
      - emetta restore sui compromessi (con priorità BV)
      - emetta reconnect sugli isolati sani
      - non emetta mai azioni invalide
      - fallback su DoNothing senza info
    """

    def test_no_info_returns_do_nothing(self, env):
        """Senza info, fallback neutro."""
        policy = OracleDefenderPolicy()
        obs = env._get_obs()
        assert policy.select_action(obs, info=None) == ACTION_DO_NOTHING

    def test_empty_compromised_returns_do_nothing(self, clean_env):
        """Rete sana e info corretto → niente da fare."""
        obs = clean_env._get_obs()
        info = {"compromised_nodes": []}
        policy = OracleDefenderPolicy()
        assert policy.select_action(obs, info) == ACTION_DO_NOTHING

    def test_restore_on_compromised_user(self, clean_env):
        """Singolo compromesso → restore di quel nodo."""
        force_state(clean_env, 3, compromised=True)
        obs = clean_env._get_obs()
        info = {"compromised_nodes": [3]}
        policy = OracleDefenderPolicy()
        assert policy.select_action(obs, info) == restore(3)

    def test_priority_critical_first(self, clean_env):
        """User (BV=1) + critical (BV=6) compromessi → restore(7) prima."""
        force_state(clean_env, 0, compromised=True)
        force_state(clean_env, 7, compromised=True)
        obs = clean_env._get_obs()
        info = {"compromised_nodes": [0, 7]}
        policy = OracleDefenderPolicy()
        assert policy.select_action(obs, info) == restore(7)

    def test_skips_nodes_already_in_restore(self, clean_env):
        """
        Nodo 3 gia' in restore, nodo 5 compromesso e non in restore:
        la policy deve scegliere restore(5), perche' Restore sul nodo 3 sarebbe invalido.
        """
        force_state(clean_env, 3, restore_timer=2, isolated=True)
        force_state(clean_env, 5, compromised=True)
        obs = clean_env._get_obs()
        # NB: nodo 3 non compare in compromised_nodes (è in restore,
        # dove la spec dice che compromised=False una volta partito il restore)
        info = {"compromised_nodes": [5]}
        policy = OracleDefenderPolicy()
        assert policy.select_action(obs, info) == restore(5)

    def test_reconnect_isolated_clean_node(self, clean_env):
        """
        Nodo 2 isolato e SANO (non in compromised_nodes), fuori restore
        → reconnect, per recuperare capacità operativa.
        """
        force_state(clean_env, 2, isolated=True)
        obs = clean_env._get_obs()
        info = {"compromised_nodes": []}
        policy = OracleDefenderPolicy()
        assert policy.select_action(obs, info) == reconnect(2)

    def test_does_not_reconnect_isolated_compromised(self, clean_env):
        """
        Nodo 2 isolato ma compromesso. Oracle non deve riconnetterlo:
        deve selezionare restore(2).
        """
        force_state(clean_env, 2, isolated=True, compromised=True)
        obs = clean_env._get_obs()
        info = {"compromised_nodes": [2]}
        policy = OracleDefenderPolicy()
        assert policy.select_action(obs, info) == restore(2)

    def test_never_emits_invalid_action_in_rollout(self):
        """200 step di rollout reale, mai un'azione invalida."""
        from envs.incident_response_env import IncidentResponseEnv
        env = IncidentResponseEnv(seed=11)
        obs, info = env.reset(seed=11)
        policy = OracleDefenderPolicy()

        for _ in range(200):
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            assert not info["action_was_invalid"], (
                f"Azione invalida emessa: action={action}, info={info}"
            )
            if terminated or truncated:
                obs, info = env.reset()
                policy.reset()