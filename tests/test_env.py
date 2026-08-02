"""
Suite di test sistematici per IncidentResponseEnv.

Copre tutti i test minimi:
    - reset e stato iniziale
    - decodifica e validazione delle azioni
    - effetti delle azioni (apply_*)
    - tempistica del restore (attraverso step)
    - propagazione statistica dell'attaccante
    - reward (incluso doppio conteggio)
    - terminazione (clean streak, all_compromised, truncated)
    - reset di analysis_state dopo infezione
    - contenuto del dizionario info

Ogni test è deterministico: i seed sono fissati ovunque, inclusi
i test "statistici" che usano un seed dedicato e tolleranze esplicite.

Eseguire con:
    pytest tests/test_env.py             # tutti i test
    pytest tests/test_env.py -v          # output verbose
    pytest tests/test_env.py::TestReward # singola classe
"""
import numpy as np
import pytest

from envs.incident_response_env import IncidentResponseEnv


# Helper functions, chiamate direttamente nei test

def force_state(env, node_id: int, **kwargs):
    """
    Forza i campi di NodeRuntime di un nodo specifico.
    Utile per costruire stati di test controllati.

    Esempio:
        force_state(env, 3, compromised=True, isolated=True)
    """
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


# Costanti per leggibilità delle azioni
# 
# Mappatura :
#   0..7   = Analyse(i)
#   8..15  = Isolate(i)
#   16..23 = Restore(i)
#   24..31 = Reconnect(i)
#   32     = DoNothing
def ANL(i):  return i           # Analyse(i)
def ISO(i):  return 8 + i       # Isolate(i)
def RST(i):  return 16 + i      # Restore(i)
def RCN(i):  return 24 + i      # Reconnect(i)
DN = 32                          # DoNothing


# Reset & osservazione iniziale

class TestReset:
    """Verifica reset() e _get_obs()."""

    def test_obs_shape_is_40(self, env):
        obs = env._get_obs()
        assert obs.shape == (40,)

    def test_obs_dtype_is_float32(self, env):
        obs = env._get_obs()
        assert obs.dtype == np.float32

    def test_obs_values_in_unit_range(self, env):
        obs = env._get_obs()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_patient_zero_is_always_user_node(self, env_no_reset):
        """Su 50 reset, patient_zero deve essere sempre user (0..4)."""
        seen = set()
        for s in range(50):
            env_no_reset.reset(seed=s)
            pz = env_no_reset.patient_zero
            assert 0 <= pz <= 4, f"reset {s}: patient_zero={pz} non è user"
            seen.add(pz)
        # Su 50 reset con seed diversi ci aspettiamo che ogni user node esca almeno
        # una volta (il patient_zero è scelto in modo uniforme sui 5 user).
        assert seen == {0, 1, 2, 3, 4}, f"non tutti gli user appaiono: {seen}"

    def test_only_patient_zero_is_compromised(self, env):
        pz = env.patient_zero
        for i, rt in enumerate(env.runtime):
            if i == pz:
                assert rt.compromised == True, f"patient_zero {pz} non compromesso"
            else:
                assert rt.compromised == False, f"nodo {i} non doveva essere compromesso"

    def test_no_node_starts_isolated(self, env):
        for rt in env.runtime:
            assert rt.isolated == False

    def test_no_node_starts_in_restore(self, env):
        for rt in env.runtime:
            assert rt.restore_timer == 0

    def test_all_analysis_states_start_zero(self, env):
        for rt in env.runtime:
            assert rt.analysis_state == 0

    def test_initial_counters(self, env):
        assert env.current_step == 0
        assert env.clean_streak == 0

    def test_initial_info_consistency(self, env_no_reset):
        obs, info = env_no_reset.reset(seed=0)
        assert info["n_compromised"] == 1
        assert info["isolated_nodes"] == []
        assert info["restoring_nodes"] == []
        assert info["clean_streak"] == 0
        assert info["current_step"] == 0
        # patient_zero è user → bv=1
        assert info["total_compromised_value"] == 1

    def test_reset_with_same_seed_is_deterministic(self, env_no_reset):
        obs1, _ = env_no_reset.reset(seed=123)
        pz1 = env_no_reset.patient_zero
        obs2, _ = env_no_reset.reset(seed=123)
        pz2 = env_no_reset.patient_zero
        assert pz1 == pz2
        np.testing.assert_array_equal(obs1, obs2)


    def test_reset_with_different_seeds_gives_different_states(self, env_no_reset):
        """
        Seed diversi devono produrre episodi diversi, altrimenti un bug che ignora il
        seed passerebbe inosservato. Poiché due seed distinti possono per caso dare lo
        stesso patient_zero, provo più seed e verifico che compaiano almeno due
        patient_zero diversi.
        """
        patient_zeros = set()
        for s in range(10):
            env_no_reset.reset(seed=s)
            patient_zeros.add(env_no_reset.patient_zero)
        assert len(patient_zeros) >= 2, \
            f"con 10 seed diversi ho visto un solo patient_zero: {patient_zeros}"


# --- Decodifica delle azioni ---

class TestActionDecoding:
    """Verifica _decode_action()."""

    @pytest.mark.parametrize("action,expected", [
        # Analyse
        (0,  ("analyse",   0)),
        (3,  ("analyse",   3)),
        (7,  ("analyse",   7)),
        # Isolate
        (8,  ("isolate",   0)),
        (10, ("isolate",   2)),
        (15, ("isolate",   7)),
        # Restore
        (16, ("restore",   0)),
        (20, ("restore",   4)),
        (23, ("restore",   7)),
        # Reconnect
        (24, ("reconnect", 0)),
        (28, ("reconnect", 4)),
        (31, ("reconnect", 7)),
        # DoNothing
        (32, ("do_nothing", None)),
    ])
    def test_decode_action(self, env, action, expected):
        assert env._decode_action(action) == expected


# --- Validazione delle azioni ---

class TestActionValidation:
    """Verifica _is_valid_action()."""

    def test_do_nothing_always_valid(self, env):
        assert env._is_valid_action("do_nothing", None) is True

    def test_analyse_always_valid_regardless_of_state(self, env):
        """Analyse non ha precondizioni — testato in vari stati"""
        for state in [
            {},
            {"compromised": True},
            {"isolated": True},
            {"restore_timer": 2},
            {"compromised": True, "isolated": True},
        ]:
            force_all_clean(env)
            force_state(env, 0, **state)
            assert env._is_valid_action("analyse", 0) is True, \
                f"Analyse invalido in stato {state}"

    def test_isolate_valid_on_non_isolated(self, env):
        force_state(env, 0, isolated=False)
        assert env._is_valid_action("isolate", 0) is True

    def test_isolate_invalid_on_already_isolated(self, env):
        force_state(env, 0, isolated=True)
        assert env._is_valid_action("isolate", 0) is False

    def test_reconnect_valid_on_isolated_not_in_restore(self, env):
        force_state(env, 0, isolated=True, restore_timer=0)
        assert env._is_valid_action("reconnect", 0) is True

    def test_reconnect_invalid_on_non_isolated(self, env):
        force_state(env, 0, isolated=False)
        assert env._is_valid_action("reconnect", 0) is False

    def test_reconnect_invalid_on_node_in_restore(self, env):
        force_state(env, 0, isolated=True, restore_timer=2)
        assert env._is_valid_action("reconnect", 0) is False

    def test_restore_valid_on_node_not_in_restore(self, env):
        force_state(env, 0, restore_timer=0)
        assert env._is_valid_action("restore", 0) is True

    def test_restore_valid_on_clean_node(self, env):
        """Restore su nodo sano è valida ma costosa, non invalida."""
        force_state(env, 0, compromised=False, isolated=False, restore_timer=0)
        assert env._is_valid_action("restore", 0) is True

    def test_restore_invalid_on_node_in_restore(self, env):
        force_state(env, 0, restore_timer=2)
        assert env._is_valid_action("restore", 0) is False


# Effetti delle azioni

class TestActionEffects:
    """Verifica _apply_*() — semantica locale delle azioni"""

    # Analyse

    def test_analyse_clean_node_sets_state_1(self, env):
        force_state(env, 2, compromised=False)
        env._apply_analyse(2)
        assert env.runtime[2].analysis_state == 1

    def test_analyse_compromised_node_sets_state_2(self, env):
        force_state(env, 3, compromised=True)
        env._apply_analyse(3)
        assert env.runtime[3].analysis_state == 2

    def test_analyse_does_not_modify_compromised(self, env):
        force_state(env, 3, compromised=True)
        before = env.runtime[3].compromised
        env._apply_analyse(3)
        assert env.runtime[3].compromised == before

    # Isolate

    def test_isolate_sets_isolated_true(self, env):
        force_state(env, 0, compromised=False, isolated=False)
        env._apply_isolate(0)
        assert env.runtime[0].isolated is True

    def test_isolate_does_not_cure(self, env):
        """Isolate su nodo compromesso NON lo guarisce."""
        force_state(env, 1, compromised=True, isolated=False)
        env._apply_isolate(1)
        assert env.runtime[1].isolated is True
        assert env.runtime[1].compromised is True

    # Restore

    def test_restore_full_effect(self, env):
        """
        Restore deve resettare compromised, isolare,
        impostare timer a 3, cancellare alert e analysis_state.
        """
        force_state(env, 4,
                    compromised=True, isolated=False, restore_timer=0,
                    alert_active=True, analysis_state=2)
        env._apply_restore(4)
        rt = env.runtime[4]
        assert rt.compromised    is False
        assert rt.isolated       is True
        assert rt.restore_timer  == 3
        assert rt.alert_active   is False
        assert rt.analysis_state == 0

    # Reconnect

    def test_reconnect_clears_isolation(self, env):
        force_state(env, 0, isolated=True, restore_timer=0)
        env._apply_reconnect(0)
        assert env.runtime[0].isolated is False

    def test_reconnect_compromised_node_returns_compromised_online(self, env):
        """
        Un nodo compromesso e isolato, riconnesso senza restore,
        torna online ma ancora compromesso
        """
        force_state(env, 3, compromised=True, isolated=True, restore_timer=0)
        env._apply_reconnect(3)
        assert env.runtime[3].isolated is False
        assert env.runtime[3].compromised is True


# Tempistica del restore attraverso step()

class TestRestoreTimer:

    def test_full_restore_timeline_via_step(self, env):
        """
        Timeline attesa con restore_timer=3 e decremento immediato:
            step t   : applico Restore → timer 3, decremento → 2 (offline)
            step t+1 : DoNothing       → timer 2 → 1            (offline)
            step t+2 : DoNothing       → timer 1 → 0            (online)

        Test interamente attraverso step(), non manipolando lo stato.
        """
        force_all_clean(env)
        force_state(env, 0, compromised=True)

        # Step t: Restore(0)
        env.step(RST(0))
        rt = env.runtime[0]
        assert rt.compromised   is False, "dopo Restore deve essere disinfettato"
        assert rt.isolated      is True,  "dopo Restore deve essere offline"
        assert rt.restore_timer == 2,     "decremento immediato: 3 -> 2"

        # Step t+1: DoNothing
        env.step(DN)
        rt = env.runtime[0]
        assert rt.restore_timer == 1, "decremento a 1"
        assert rt.isolated      is True, "ancora offline"

        # Step t+2: DoNothing
        env.step(DN)
        rt = env.runtime[0]
        assert rt.restore_timer == 0,    "timer arrivato a 0"
        assert rt.isolated      is False, "torna online"
        assert rt.compromised   is False, "ancora pulito"
        assert rt.analysis_state == 0,    "analysis_state azzerato"


# Propagazione statistica

class TestPropagation:

    def test_p_spread_matches_specification(self):
        """
        Setup: nodo 0 compromesso e online; nodo 5 (suo unico vicino) sano.
        L'attaccante di nodo 0 ha un solo target possibile: nodo 5.
        Frequenza di infezione attesa = p_spread = 0.30.

        Su 1000 prove con seed fisso, asserzione: 0.25 <= p <= 0.35.
        Tolleranza: sigma = sqrt(0.3*0.7/1000) ~= 0.0145
        => +/- 0.05 copre la varianza statistica.
        """

        N_TRIALS = 1000
        infections = 0

        env = IncidentResponseEnv(seed=12345)
        env.reset(seed=12345)

        for _ in range(N_TRIALS):
            # Reset di setup ad ogni prova
            force_all_clean(env)
            force_state(env, 0, compromised=True)
            # Nodo 5 è l'unico vicino di nodo 0 (vedi topology), quindi
            # l'attaccante avrà sempre [5] come unica scelta.
            # DoNothing per non interferire
            env.step(DN)
            if env.runtime[5].compromised:
                infections += 1

        p_observed = infections / N_TRIALS
        assert 0.25 <= p_observed <= 0.35, \
            f"p_spread osservata = {p_observed:.4f}, attesa ~0.30 (range 0.25-0.35)"


# --- Reward ---

class TestReward:
    """Verifica la formula della reward, incluso il doppio conteggio."""

    def test_reward_clean_state_is_zero(self, env):
        """Stato perfettamente sano e DoNothing → reward = 0"""
        force_all_clean(env)
        _, reward, _, _, _ = env.step(DN)
        assert reward == 0.0

    def test_reward_compromised_isolated_double_count(self, env):
        """
        Stato: nodo 0 (user, v=1) compromesso E isolato (non in restore),
        gli altri sani e online.
        Reward attesa: -alpha*1 (C_t) - beta*1 (I_t) = -1.5
        Conferma il doppio conteggio.
        """
        force_all_clean(env)
        force_state(env, 0, compromised=True, isolated=True, restore_timer=0)
        _, reward, _, _, _ = env.step(DN)
        # nodo 0 è isolato: non si propaga, gli altri restano sani
        assert abs(reward - (-1.5)) < 1e-6, f"reward={reward}, atteso -1.5"

    def test_reward_isolated_only(self, env):
        """
        Stato: nodo 5 (service, v=3) isolato sano (non in restore),
        gli altri sani e online.
        Reward attesa: -beta*3 = -1.5
        """
        force_all_clean(env)
        force_state(env, 5, isolated=True, compromised=False, restore_timer=0)
        _, reward, _, _, _ = env.step(DN)
        assert abs(reward - (-1.5)) < 1e-6, f"reward={reward}, atteso -1.5"

    def test_reward_restoring_only(self, env):
        """
        Stato: nodo 7 (critical, v=6) in restore (compromised=False, timer=2),
        gli altri sani.
        Reward attesa: -gamma*6 = -6.0 (NON conta in I_t per regola del doppio
        conteggio).
        """
        force_all_clean(env)
        force_state(env, 7, compromised=False, isolated=True, restore_timer=2)
        _, reward, _, _, _ = env.step(DN)
        # Dopo decremento timer è 1 (ancora in restore)
        assert env.runtime[7].restore_timer == 1
        assert abs(reward - (-6.0)) < 1e-6, f"reward={reward}, atteso -6.0"

    def test_reward_invalid_action_penalty(self, env):
        """
        Penalità per azione invalida = -eta = -0.25.
        Setup: nodo 0 isolato. Tento Isolate(0) → invalida.
        Reward = -beta*1 (per l'isolamento) - 0.25 (penalità) = -0.75
        """
        force_all_clean(env)
        force_state(env, 0, isolated=True)
        _, reward, _, _, info = env.step(ISO(0))
        assert info["action_was_invalid"] is True
        assert abs(reward - (-0.75)) < 1e-6, f"reward={reward}, atteso -0.75"

    def test_reward_critical_compromised_full_cost(self, env):
        """
        Nodo critical (v=6) compromesso e isolato:
        reward = -alpha*6 - beta*6 = -9.0
        Verifica che il business value sia rispettato.
        """
        force_all_clean(env)
        force_state(env, 7, compromised=True, isolated=True, restore_timer=0)
        _, reward, _, _, _ = env.step(DN)
        assert abs(reward - (-9.0)) < 1e-6, f"reward={reward}, atteso -9.0"


# Terminazione e clean streak

class TestTermination:

    def test_terminated_after_3_clean_steps(self, env):
        """Da rete pulita, 3 DoNothing consecutivi → terminated al 3°."""
        force_all_clean(env)

        # Step 1 e 2: clean_streak cresce ma NON termina ancora
        for n in range(1, 3):
            _, _, term, _, info = env.step(DN)
            assert term is False, f"step {n}: terminated troppo presto"
            assert info["clean_streak"] == n

        # Step 3: terminated scatta
        _, _, term, _, info = env.step(DN)
        assert term is True, "deve terminare al 3° clean step"
        assert info["clean_streak"] == 3

    def test_clean_streak_reset_by_compromised(self, env):
        """Un nodo compromesso (anche isolato) azzera il clean_streak"""
        force_all_clean(env)
        env.step(DN)
        assert env.clean_streak == 1
        # Forza compromissione ma ISOLA per non far propagare
        force_state(env, 1, compromised=True, isolated=True)
        env.step(DN)
        assert env.clean_streak == 0

    def test_clean_streak_reset_by_restoring(self, env):
        """
        Un nodo in restore azzera il clean_streak anche se non c'è
        nessun nodo compromesso
        """
        force_all_clean(env)
        env.step(DN)
        assert env.clean_streak == 1
        # Forza un nodo in restore
        force_state(env, 0, compromised=False, isolated=True, restore_timer=2)
        env.step(DN)
        assert env.clean_streak == 0

    def test_truncated_after_max_steps(self, env):
        """Dopo 40 step di episodio non terminato → truncated."""
        force_all_clean(env)
        # Tengo un nodo compromesso e isolato per evitare clean_streak e
        # impedire la propagazione (così non si arriva a terminated)
        force_state(env, 0, compromised=True, isolated=True)

        terminated = False
        truncated  = False
        steps      = 0

        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(DN)
            steps += 1
            if steps > 100:
                pytest.fail("loop infinito — qualcosa non va")

        assert truncated is True
        assert info["current_step"] == 40

    def test_terminated_when_all_compromised(self, env):
        """
        Tutti i nodi compromessi → terminated.
        Verifica anche che lo stato finale sia coerente con la causa
        della terminazione.
        """
        force_all_clean(env)
        for i in range(8):
            force_state(env, i, compromised=True)
        _, _, term, _, info = env.step(DN)
        assert term is True
        # Verifica esplicita della causa: tutti i nodi devono essere
        # ancora compromessi a fine step (non un altro motivo)
        assert all(rt.compromised for rt in env.runtime), \
            "terminated è scattato ma non per all_compromised"
        assert info["n_compromised"] == 8


# --- Reset di analysis_state dopo infezione ---

class TestAnalysisStateResetAfterInfection:
    """
    Il reset di analysis_state avviene davvero
    attraverso step(), non manipolando lo stato manualmente.
    """

    def test_analysis_state_resets_when_infected_through_step(self):
        """
        Setup:
            - nodo 0: noto sano (analysis_state=1), online
            - nodo 5: compromesso e online (suo unico vicino è 0, 1, 4, 7)
            - nodi 1, 4, 7: isolati per essere fuori dalla lista degli
              infettabili → l'attaccante di nodo 5 ha solo nodo 0 come target

        Eseguo step(DoNothing) finché il nodo 0 non viene infettato.
        Verifica: dopo l'infezione, analysis_state[0] == 0.
        """
        env = IncidentResponseEnv(seed=42)
        env.reset(seed=42)
        force_all_clean(env)

        force_state(env, 0,
                    compromised=False, isolated=False,
                    restore_timer=0, analysis_state=1)
        force_state(env, 5,
                    compromised=True, isolated=False,
                    restore_timer=0)
        # Isolo gli altri vicini di 5 per assicurare che 0 sia
        # l'unico target infettabile dell'attaccante.
        for nb in (1, 4, 7):
            force_state(env, nb, isolated=True)

        infected = False
        for _ in range(200):
            env.step(DN)
            if env.runtime[0].compromised:
                infected = True
                break

        assert infected, ("nodo 0 mai infettato in 200 step "
                          "(probabilità < 1e-30, qualcosa è rotto)")
        assert env.runtime[0].analysis_state == 0, \
            f"analysis_state non resettato: {env.runtime[0].analysis_state}"


# --- Contenuto del dizionario info ---

class TestInfoDict:

    REQUIRED_FIELDS = (
        "compromised_nodes", "isolated_nodes", "restoring_nodes",
        "n_compromised", "total_compromised_value",
        "action_was_invalid", "action_type", "action_target",
        "clean_streak", "current_step",
    )

    def test_info_contains_required_fields_after_reset(self, env_no_reset):
        _, info = env_no_reset.reset(seed=0)
        for field in self.REQUIRED_FIELDS:
            assert field in info, f"campo mancante dopo reset: {field}"

    def test_info_contains_required_fields_after_step(self, env):
        _, _, _, _, info = env.step(DN)
        for field in self.REQUIRED_FIELDS:
            assert field in info, f"campo mancante dopo step: {field}"

    def test_info_action_target_for_donothing_is_none(self, env):
        _, _, _, _, info = env.step(DN)
        assert info["action_type"] == "do_nothing"
        assert info["action_target"] is None

    def test_info_action_target_for_invalid_action_keeps_target(self, env):
        """
        Per le azioni invalide manteniamo il target del nodo bersagliato
        """
        force_all_clean(env)
        force_state(env, 3, isolated=True)
        # Isolate(3) su nodo già isolato → invalida
        _, _, _, _, info = env.step(ISO(3))
        assert info["action_was_invalid"] is True
        assert info["action_type"]       == "invalid"
        assert info["action_target"]     == 3   # target conservato

    def test_info_action_type_for_valid_actions(self, env):
        """Verifica i nomi delle azioni nel campo info per ogni tipo."""
        force_all_clean(env)

        # Analyse(2)
        _, _, _, _, info = env.step(ANL(2))
        assert info["action_type"] == "analyse"
        assert info["action_target"] == 2
        assert info["action_was_invalid"] is False

        # Isolate(2) — ripartiamo da stato pulito per indipendenza tra blocchi
        force_all_clean(env)
        _, _, _, _, info = env.step(ISO(2))
        assert info["action_type"] == "isolate"
        assert info["action_target"] == 2

        # Restore(2)
        force_all_clean(env)
        _, _, _, _, info = env.step(RST(2))
        assert info["action_type"] == "restore"
        assert info["action_target"] == 2

        # Reconnect(2)
        force_all_clean(env)
        force_state(env, 2, isolated=True, restore_timer=0)
        _, _, _, _, info = env.step(RCN(2))
        assert info["action_type"] == "reconnect"
        assert info["action_target"] == 2

    def test_info_action_type_original_preserves_decoded_type_on_invalid(self, env):
        """
        Per le azioni invalide:
            info["action_type"]          deve essere "invalid"
            info["action_type_original"] deve preservare il tipo decodificato
        """
        # Isolate su nodo già isolato → invalid, original = "isolate"
        force_all_clean(env)
        force_state(env, 3, isolated=True)
        _, _, _, _, info = env.step(ISO(3))
        assert info["action_was_invalid"] is True
        assert info["action_type"] == "invalid"
        assert info["action_type_original"] == "isolate"

        # Restore su nodo già in restore → invalid, original = "restore"
        force_all_clean(env)
        force_state(env, 4, restore_timer=2, isolated=True)
        _, _, _, _, info = env.step(RST(4))
        assert info["action_was_invalid"] is True
        assert info["action_type"] == "invalid"
        assert info["action_type_original"] == "restore"

        # Reconnect su nodo non isolato → invalid, original = "reconnect"
        force_all_clean(env)
        _, _, _, _, info = env.step(RCN(5))
        assert info["action_was_invalid"] is True
        assert info["action_type"] == "invalid"
        assert info["action_type_original"] == "reconnect"

    def test_info_action_type_original_matches_action_type_on_valid(self, env):
        """
        Sulle azioni valide, action_type_original deve coincidere con
        action_type (non c'è differenza diagnostica da preservare).
        """
        force_all_clean(env)
        _, _, _, _, info = env.step(ANL(0))
        assert info["action_type"] == "analyse"
        assert info["action_type_original"] == "analyse"

        _, _, _, _, info = env.step(DN)
        assert info["action_type"] == "do_nothing"
        assert info["action_type_original"] == "do_nothing"

    def test_info_total_compromised_value(self, env):
        """Verifica il calcolo del business value totale dei compromessi."""
        force_all_clean(env)
        force_state(env, 0, compromised=True, isolated=True)  # user, v=1
        force_state(env, 5, compromised=True, isolated=True)  # service, v=3
        _, _, _, _, info = env.step(DN)
        assert info["n_compromised"] == 2
        assert info["total_compromised_value"] == 4   # 1 + 3

    def test_info_isolated_nodes_excludes_restoring(self, env):
        """
        isolated_nodes contiene solo i nodi isolati e NON in restore.
        Un nodo in restore deve apparire solo in restoring_nodes.
        """
        force_all_clean(env)
        force_state(env, 0, isolated=True, restore_timer=0)  # solo isolato
        force_state(env, 1, isolated=True, restore_timer=2)  # in restore
        _, _, _, _, info = env.step(DN)
        assert 0 in info["isolated_nodes"]
        assert 0 not in info["restoring_nodes"]
        assert 1 in info["restoring_nodes"]
        assert 1 not in info["isolated_nodes"]



# --- Re-infezione post-restore nello stesso step ---

class TestReinfectionAfterRestore:
    """
    Verifica la dinamica di reinfezione nello stesso step: se il restore_timer arriva a 0 
    e il nodo torna online, è subito soggetto alla propagazione dello stesso step 
    e può essere reinfettato.

    Con un setup controllato e molti tentativi l'evento si verifica quasi certamente almeno una volta.
    """

    def test_node_can_be_reinfected_after_restore_completes_same_step(self):
        """
        Setup ripetuto N volte:
            - nodo 5 (service) compromesso e online
            - nodo 0 con restore_timer=1 (uscirà al prossimo step)
              e analysis_state=0
            - nodi 1, 4, 7 isolati per forzare 0 come unico target
              di propagazione di 5

        Ad ogni ripetizione si esegue un solo step(DN):
            - timer di 0: 1 → 0, nodo torna online (isolated=False)
            - propagazione di 5 vede 0 come unico target infettabile
            - con p_spread=0.30, prob. di reinfezione per step ≈ 0.30

        Su 100 ripetizioni la probabilità che 0 non venga mai
        reinfettato è (1 - 0.30)^100 ≈ 3 * 10^-16, quindi trascurabile.
        """
        env = IncidentResponseEnv(seed=42)
        env.reset(seed=42)

        reinfections_observed = 0
        N_TRIALS = 100

        for _ in range(N_TRIALS):
            # Setup deterministico per questa prova
            force_all_clean(env)
            force_state(env, 0,
                        compromised=False, isolated=True,
                        restore_timer=1, analysis_state=0)
            force_state(env, 5,
                        compromised=True, isolated=False,
                        restore_timer=0)
            # Isolo 1, 4, 7 per garantire che 0 sia l'unico target
            # infettabile dell'attaccante di 5 (vicini di 5: 0, 1, 4, 7)
            for nb in (1, 4, 7):
                force_state(env, nb, isolated=True)

            # Prima dello step non deve esserci alcun target infettabile.
            attacker_neighbors = env.network[5].neighbors
            infectable_targets = [
                nb for nb in attacker_neighbors
                if not env.runtime[nb].compromised
                and not env.runtime[nb].isolated
                and env.runtime[nb].restore_timer == 0
            ]
            # Il nodo 0 diventa infettabile solo dopo il decremento del restore_timer.
            assert 0 not in infectable_targets, (
                "setup errato: nodo 0 risulta infettabile prima dello step"
            )
            assert len(infectable_targets) == 0, (
                f"setup errato: l'attaccante ha {len(infectable_targets)} "
                f"target validi prima dello step (atteso 0). "
                f"Probabile modifica della topologia: rivedere il test."
            )


            env.step(DN)

            # Verifica le proprietà del singolo step
            rt0 = env.runtime[0]
            # Il timer deve essere arrivato a 0 e il nodo essere online
            assert rt0.restore_timer == 0, \
                "timer non arrivato a 0 nel singolo step"
            assert rt0.isolated is False, \
                "nodo 0 doveva tornare online dopo decremento del timer"

            # Se è compromesso, è stata una reinfezione nello stesso step
            if rt0.compromised:
                reinfections_observed += 1

        # Su 100 prove con p=0.30 ci aspettiamo ~30 reinfezioni.
        # Asserzione: almeno una reinfezione è avvenuta
        assert reinfections_observed > 0, (
            f"in {N_TRIALS} prove con p_spread=0.30 nessuna reinfezione "
            f"nello stesso step: la dinamica di reinfezione post-restore non funziona"
        )
        # Range statistico ragionevole: tra 15 e 50 (3σ ≈ ±14 attorno a 30)
        assert 15 <= reinfections_observed <= 50, (
            f"frequenza reinfezione anomala: {reinfections_observed}/100 "
            f"(atteso ~30 ± 14)"
        )