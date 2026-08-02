"""
Policy passiva: emette sempre DoNothing (Limite inferiore di prestazione)
"""
from baselines.policy_base import BasePolicy, ACTION_DO_NOTHING


class DoNothingPolicy(BasePolicy):
    name = "do_nothing"

    def select_action(self, obs, info=None) -> int:
        return ACTION_DO_NOTHING