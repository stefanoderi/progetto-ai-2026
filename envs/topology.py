"""Modellazione statica dei nodi e della topologia di rete a 8 nodi"""

from dataclasses import dataclass


NODE_TYPE_USER     = "user"
NODE_TYPE_SERVICE  = "service"
NODE_TYPE_CRITICAL = "critical"

BUSINESS_VALUE = {
    NODE_TYPE_USER:     1,
    NODE_TYPE_SERVICE:  3,
    NODE_TYPE_CRITICAL: 6,
}


@dataclass(frozen=True)
class NodeSpec:
    """Parte statica e immutabile di un nodo"""
    node_id:        int
    node_type:      str
    business_value: int
    neighbors:      tuple   # tuple, non lista: frozen=True vieta campi mutabili


@dataclass
class NodeRuntime:
    """Stato dinamico del nodo, resettato a ogni episodio"""
    compromised:    bool = False
    isolated:       bool = False
    restore_timer:  int  = 0   # 0 = non in restore; >0 = step rimanenti al completamento
    alert_active:   bool = False
    analysis_state: int  = 0   # 0 = sconosciuto, 1 = noto sano, 2 = noto compromesso


def build_network() -> list[NodeSpec]:
    """
    Costruisce la rete: nodi 0-4 user (BV 1), 5-6 service (BV 3), 7 critical (BV 6).
    Gli user sono periferici, i service fanno da ponte verso il nodo critico
    """
    adjacency = {
        0: (5,),
        1: (5,),
        2: (6,),
        3: (6,),
        4: (5, 6),
        5: (0, 1, 4, 7),
        6: (2, 3, 4, 7),
        7: (5, 6),
    }

    node_types = {
        0: NODE_TYPE_USER,
        1: NODE_TYPE_USER,
        2: NODE_TYPE_USER,
        3: NODE_TYPE_USER,
        4: NODE_TYPE_USER,
        5: NODE_TYPE_SERVICE,
        6: NODE_TYPE_SERVICE,
        7: NODE_TYPE_CRITICAL,
    }

    specs = []
    for node_id in range(8):
        ntype = node_types[node_id]
        specs.append(NodeSpec(
            node_id=node_id,
            node_type=ntype,
            business_value=BUSINESS_VALUE[ntype],
            neighbors=adjacency[node_id],
        ))

    return specs