"""
Fixture pytest condivise dalla suite di test dell'ambiente. Gli helper
force_state/force_all_clean stanno in test_env.py.
"""
import pytest

from envs.incident_response_env import IncidentResponseEnv


@pytest.fixture
def env():
    """
    Ambiente pulito con reset() gia' chiamato (seed=0), punto di partenza standard.
    Scope 'function' di default: ogni test riceve un'istanza nuova e indipendente.
    """
    e = IncidentResponseEnv(seed=0)
    e.reset(seed=0)
    return e


@pytest.fixture
def env_no_reset():
    """Ambiente NON resettato (solo __init__), per i test che chiamano reset() esplicitamente."""
    return IncidentResponseEnv(seed=0)
