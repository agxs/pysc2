"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import pickle

import neat
import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


runs_per_net = 5
simulation_seconds = 60.0

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class NeatAgent(base_agent.BaseAgent):

    def setup(self, obs_spec, action_spec):
        super(NeatAgent, self).setup(obs_spec, action_spec)
        self.selected_army = False
        self.initial_move = False

    def reset(self):
        super(NeatAgent, self).reset()
        self.selected_army = False
        self.initial_move = False

    def step(self, obs, net, genome):
        super(NeatAgent, self).step(obs)

        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        selected = obs.observation["screen"][_SELECTED]

        action = net.activate(list(player_relative.flat) + list(selected.flat))
        if action[2] < 0.0:
            action[2] = 0.0
        if action[2] > 1.0:
            action[2] = 1.0
        if action[3] < 0.0:
            action[3] = 0.0
        if action[3] > 1.0:
            action[3] = 1.0

        if action[0] > 0.5:
            if not self.selected_army:
                genome.fitness = genome.fitness + 50.0
                self.selected_army = True
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        elif action[1] > 0.5:
            # print("trying a move")
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                target = [int(action[2] * 15.0), int(action[3] * 15.0)]
                # print("Moving to target! ", target)
                if not self.initial_move:
                    genome.fitness = genome.fitness + 25.0
                    self.initial_move = True
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
            else:
                # print("No selection")
                return actions.FunctionCall(_NO_OP, [])
        else:
            return actions.FunctionCall(_NO_OP, [])
