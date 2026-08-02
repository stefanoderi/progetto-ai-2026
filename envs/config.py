"""Configurazione globale dell'ambiente IncidentResponseEnv"""
from dataclasses import dataclass


@dataclass
class EnvConfig:

    # Rete
    n_nodes: int = 8

    # Dinamica attaccante
    p_spread: float = 0.30

    # Osservazione
    p_alert_true: float = 0.70    # P(alert | nodo compromesso)
    p_alert_false: float = 0.10   # P(alert | nodo sano) → falso positivo

    # Restore
    restore_timer_init: int = 3   # il nodo resta offline 2 step effettivi: il decremento
                                  # avviene nello stesso step dell'azione

    # Episodio
    max_steps: int = 40
    clean_streak_target: int = 3  # step puliti consecutivi per terminated (vittoria)

    # Coefficienti reward
    alpha: float = 1.0    # peso nodi compromessi
    beta: float  = 0.5    # peso nodi isolati (non in restore)
    gamma: float = 1.0    # peso nodi in restore
    eta: float   = 0.25   # penalita' azione invalida

    # Action space
    n_nodes_actions: int = 8   # azioni per tipo = n_nodes
    
    # totale azioni = 8 * 4 + 1 = 33