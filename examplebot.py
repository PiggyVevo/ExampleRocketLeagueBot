from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
import os
import json
import socket
from typing import Dict, Any

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
        
project_name="ExampleBot"

from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np

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
        (InAirReward(), 0.15),
        (SpeedTowardBallReward(), 5),
        (VelocityBallToGoalReward(), 10),
        (TouchReward(), 50),
        (GoalReward(), 500.0)
    )

    obs_builder = DefaultObs(zero_padding=3,
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                              1 / common_values.BACK_NET_Y, 
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                           boost_coef=1 / 100.0)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer()
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_files = os.listdir(checkpoint_folder)
    checkpoint_load_folder = os.path.join(checkpoint_folder, max(checkpoint_files)) if checkpoint_files else None


    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None, # Leave this empty for now.
                      ppo_batch_size=100_000,  # batch size - much higher than 300K doesn't seem to help most people
                      policy_layer_sizes=[512, 512, 512],  # policy network
                      critic_layer_sizes=[512, 512, 512],  # critic network
                      ts_per_iteration=100_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=300_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,
                      render=True,
                      render_delay=0.047,
                      add_unix_timestamp=False,
                      checkpoint_load_folder=checkpoint_load_folder,
                      checkpoints_save_folder=checkpoint_folder,                      # entropy coefficient - this determines the impact of exploration
                      policy_lr=1e-4,
                      device="cpu", # policy learning rate
                      critic_lr=1e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=1_000_000,  # save every 1M steps
                      timestep_limit=1_000_000_000,  # Train for 1B steps
                      log_to_wandb=False # Set this to True if you want to use Weights & Biases for logging.
                      ) 
    learner.learn()
