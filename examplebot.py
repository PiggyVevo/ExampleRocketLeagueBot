from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
import os
from typing import Dict, Any


project_name="ExampleBot" #the name of your bot, changing this will start a new run 

#=========================================
#Renderer
#=========================================
import json
import socket
import numpy as np
from rlgym.api import Renderer
from rlgym.rocket_league.api import GameState, Car

DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 9273  # Default RocketSimVis port
BUTTON_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")

class RocketSimVisRenderer(Renderer[GameState]):
    """
    A renderer that sends game state information to RocketSimVis.

    This is just the client side, you need to run RocketSimVis to see the visualization.
    Code is here: https://github.com/ZealanL/RocketSimVis
    """
    def __init__(self, udp_ip=DEFAULT_UDP_IP, udp_port=DEFAULT_UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.udp_ip = udp_ip
        self.udp_port = udp_port

    @staticmethod
    def write_physobj(physobj):
        j = {
            'pos': physobj.position.tolist(),
            'forward': physobj.forward.tolist(),
            'up': physobj.up.tolist(),
            'vel': physobj.linear_velocity.tolist(),
            'ang_vel': physobj.angular_velocity.tolist()
        }

        return j

    @staticmethod
    def write_car(car: Car, controls=None):
        j = {
            'team_num': int(car.team_num),
            'phys': RocketSimVisRenderer.write_physobj(car.physics),
            'boost_amount': car.boost_amount,
            'on_ground': bool(car.on_ground),
            "has_flipped_or_double_jumped": bool(car.has_flipped or car.has_double_jumped),
            'is_demoed': bool(car.is_demoed),
            'has_flip': bool(car.can_flip)
        }

        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = {
                    k: float(v)
                    for k, v in zip(BUTTON_NAMES, controls)
                }
            j['controls'] = controls

        return j

    def render(self, state: GameState, shared_info: Dict[str, Any]) -> Any:
        if "controls" in shared_info:
            controls = shared_info["controls"]
        else:
            controls = {}
        j = {
            'ball_phys': self.write_physobj(state.ball),
            'cars': [
                self.write_car(car, controls.get(agent_id))
                for agent_id, car in state.cars.items()
            ],
            'boost_pad_states': (state.boost_pad_timers <= 0).tolist()
        }

        self.sock.sendto(json.dumps(j).encode('utf-8'), (self.udp_ip, self.udp_port))

    def close(self):
        pass

#=========================================
#State Setters
#=========================================

from typing import Dict, Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from rlgym.rocket_league.math import rand_vec3, rand_uvec3, normalize

from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import RAMP_HEIGHT


class RandomPhysicsMutator(StateMutator[GameState]):  #taken from rlgym tools, slightly modified
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 100  # Ball radius and car hitbox with biggest radius are both below this
        goal_line_y = 5120
        min_goal_dist = 2000
        i = 0

        for po in [state.ball] + [car.physics for car in state.cars.values()]:
            while True:
                if i == 0:
                    max_z = CEILING_Z - padding
                else:
                    # Cars spawn at max 1/6 ceiling height, because falling from the sky is pointless
                    max_z = (CEILING_Z / 6) - padding

                new_pos = np.random.uniform(
                    [-SIDE_WALL_X + padding, -BACK_WALL_Y + padding, 0 + padding],
                    [SIDE_WALL_X - padding, BACK_WALL_Y - padding, max_z]
                )

               #Make sure ball spawns at least 2000 uu from both goal lines
                if i == 0 and (abs(new_pos[1]) > goal_line_y - min_goal_dist):
                    continue

                # Field edge checks
                if abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - padding:
                    continue

                close_to_wall = (
                    abs(new_pos[0]) >= SIDE_WALL_X - RAMP_HEIGHT or
                    abs(new_pos[1]) >= BACK_WALL_Y - RAMP_HEIGHT or
                    abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - RAMP_HEIGHT
                )
                close_to_floor_or_ceiling = (
                    new_pos[2] <= RAMP_HEIGHT or
                    new_pos[2] >= CEILING_Z - RAMP_HEIGHT
                )

                if close_to_wall and close_to_floor_or_ceiling:
                    continue

                break

            # Assign position and random motion
            po.position = new_pos
            po.linear_velocity = rand_vec3(2300)
            po.angular_velocity = rand_vec3(5)

            # Set rotation matrix for cars only
            if i > 0:
                fw = rand_uvec3()
                up = rand_uvec3()
                right = normalize(np.cross(up, fw))
                up = normalize(np.cross(fw, right))
                rot_mat = np.stack([fw, right, up])
                po.rotation_mtx = rot_mat

            i += 1
            
from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import VariableTeamSizeMutator
        
from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import WeightedSampleMutator
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator
    
class RandomStateMutator(StateMutator[GameState]):
    def __init__(self):
        self.mutator = WeightedSampleMutator.from_zipped(
            (KickoffMutator(), 0.6),  #this means that 60% of the time, the ball and the cars will be in kickoff positions
            (RandomPhysicsMutator(), 0.4)   #this means that 40% of the time, the ball and the cars will be in random positions         
        )

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self.mutator.apply(state, shared_info)

#=========================================
#Rewards
#=========================================
        
from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np

from typing import Any, Dict, List
import numpy as np
from rlgym.rocket_league.common_values import BALL_MAX_SPEED

class AdvancedTouchReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, touch_reward: float = 0.0, acceleration_reward: float = 1, use_touch_count: bool = False):
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.use_touch_count = use_touch_count

        self.prev_ball = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball = initial_state.ball

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0 for agent in agents}
        ball = state.ball
        for agent in agents:
            touches = state.cars[agent].ball_touches

            if touches > 0:
                if not self.use_touch_count:
                    touches = 1
                acceleration = np.linalg.norm(ball.linear_velocity - self.prev_ball.linear_velocity) / BALL_MAX_SPEED
                rewards[agent] += self.touch_reward * touches
                rewards[agent] += acceleration * self.acceleration_reward

        self.prev_ball = ball

        return rewards

class FaceBallReward(RewardFunction):
    """Rewards the agent for facing the ball"""
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass


    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            car_pos = car.physics.position
            ball_pos = ball.position
            direction_to_ball = ball_pos - car_pos
            norm = np.linalg.norm(direction_to_ball)

            if norm > 0:
                direction_to_ball /= norm

            car_forward = car.physics.forward
            dot_product = np.dot(car_forward, direction_to_ball)

            reward = dot_product  # Dot product directly indicates alignment (-1 to 1)
            rewards[agent] = reward

        return rewards
                        
class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = (ball_physics.position - car_physics.position)
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
        return rewards

class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)
        return rewards


class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward of 1 if the agent touches the ball, 0 otherwise.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        return 1. if state.cars[agent].ball_touches > 0 else 0.


#=========================================
#Training Script
#=========================================
def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(  
        (InAirReward(), 0.15), #bots hate jumping/doing aerials, so we need reeards to get the bot to jump, for aerial touch rewards, you need lots of encouragement for it to aerial
        (SpeedTowardBallReward(), 5.0), #for driving towards the ball
        (FaceBallReward(), 1.0), #for facing the ball so bot doesn't drive backwards
        (VelocityBallToGoalReward(), 10.0), #for encouraging agents to hit the ball towards the opponents goal
        (AdvancedTouchReward(touch_reward=0.5, acceleration_reward=1.0), 75.0), #for strong touches, not just soft touches
        (GoalReward(), 500.0) #scoring,don't set this too high though, as it will drown out other rewards
    )
    #the rewards listed above are just sample rewards(the weights are pretty bad), follow this tutorial for more information: https://www.youtube.com/watch?v=l3j8-re_x7Q
    
    obs_builder = DefaultObs(zero_padding=3,
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                              1 / common_values.BACK_NET_Y, 
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                           boost_coef=1 / 100.0) #your observation builder, how your bot perceives the game

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        RandomStateMutator() #RandomStateMutator is better because having the ball be randomly spawning in some spots helps with exploration
    )  

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer() #our renderer
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    # 32 processes
    n_proc = 32 #set this as high as you can go without lagging a ton when you start up training, I have an i7-12700k cpu, and I use 96 processes
                #the better cpu you have, the higher you set it for max steps per second, this setting does not matter how good of a gpu you have or if you even have a gpu(like me)

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    #Our code for loading our checkpoints
    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    checkpoint_files = os.listdir(checkpoint_folder)
    checkpoint_load_folder = os.path.join(checkpoint_folder, max(checkpoint_files)) if checkpoint_files else None


    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None, # Leave this empty for now.
                      ppo_batch_size=100_000,  # batch size - much higher than 300K doesn't seem to help most people, increase this as you go
                      policy_layer_sizes=[512, 512, 512],  # policy network
                      critic_layer_sizes=[512, 512, 512],  # critic network
                      ts_per_iteration=100_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=300_000,  # size of experience buffer - keep this 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle, if you run on cpu, set it to something very big (RAM is usually bigger than VRAM), or something quite small (CPU cache size)
                      ppo_ent_coef=0.01, #0.01 is pretty optimal, this determines the impact of exploration
                      render=True, #Keep this on for rendering, just make sure not to leave the renderer window open in rocketsimvis
                      n_checkpoints_to_keep=1000, #set this to as high as you want, this is just how many checkpoints you keep
                      render_delay=0.047, #this will visualize the game at around normal speed, increasing this will make the game go slower, decreasing this will make the game go faster
                      add_unix_timestamp=False, #disable this, otherwise checkpoints will not load
                      checkpoint_load_folder=checkpoint_load_folder,
                      checkpoints_save_folder=checkpoint_folder,                      
                      policy_lr=2e-4, # policy learning rate
                      device="auto", #device to use
                      critic_lr=2e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs(how many times the learner looks over the data), 2 is pretty optimal for best steps per second and learnign quality
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=10_000_000,  # save every 10M steps
                      timestep_limit=50_000_000_000,  # Train for 50B steps
                      log_to_wandb=False # Set this to True if you want to use Weights & Biases for logging.
                      ) 
    learner.learn() #we start training!
