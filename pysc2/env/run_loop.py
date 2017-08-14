# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import os
import pickle

import neat
import numpy
import math

import time

g_max_frames = 0

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()
  g_max_frames = max_frames

  action_spec = env.action_spec()
  observation_spec = env.observation_spec()
  for agent in agents:
    agent.setup(observation_spec, action_spec)



  # Load the config file, which is assumed to live in
  # the same directory as this script.
  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, 'config-neat')
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_path)

  pop = neat.Population(config)
  stats = neat.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.StdOutReporter(True))

  config.sc2 = {}
  config.sc2['env'] = env
  config.sc2['agents'] = agents
  config.sc2['total_frames'] = 0

  # pe = neat.ParallelEvaluator(4, eval_genome)
  # winner = pop.run(pe.evaluate)
  winner = pop.run(eval_genomes)

  # Save the winner.
  with open('winner-movetobeacon', 'wb') as f:
    pickle.dump(winner, f)

  print(winner)

  node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}

  elapsed_time = time.time() - start_time
  print("Took %.3f seconds for %s steps: %.3f fps" % (
    elapsed_time, config.sc2['total_frames'], config.sc2['total_frames'] / elapsed_time))

def eval_genomes(genomes, config):
  env = config.sc2['env']
  agents = config.sc2['agents']

  try:
    for genome_id, genome in genomes:

      print("new genome")

      genome.fitness = 0.0
      timesteps = env.reset()
      net = neat.nn.FeedForwardNetwork.create(genome, config)
      for a in agents:
        a.reset()

      timestep = timesteps[0]

      player_relative = timestep.observation["screen"][_PLAYER_RELATIVE]
      original_y, original_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      original = [int(original_x.mean()), int(original_y.mean())]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      target = [int(neutral_x.mean()), int(neutral_y.mean())]

      vec = [target[0] - original[0], target[1] - original[1]]
      orig_distance = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])

      while True:
        config.sc2['total_frames'] += 1
        actions_r = [agent.step(timestep, net, genome)
                  for agent, timestep in zip(agents, timesteps)]
        if g_max_frames and config.sc2['total_frames'] >= g_max_frames:
          break
        if timesteps[0].last():
          break
        timesteps = env.step(actions_r)

      timestep = timesteps[0]

      player_relative = timestep.observation["screen"][_PLAYER_RELATIVE]

      marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      source = [int(marine_x.mean()), int(marine_y.mean())]

      vec = [target[0] - source[0], target[1] - source[1]]
      new_distance = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])

      score = (1.0 - (new_distance / orig_distance)) * 100.0
      print("Adding score: ", score)

      genome.fitness = genome.fitness + score
      print("Final score: ", genome.fitness)

  except KeyboardInterrupt:
    pass
