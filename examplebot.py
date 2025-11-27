from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
import os
import numpy as np
from rewards import FaceBallReward, SpeedTowardBallReward, InAirReward, VelocityBallToGoalReward
from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
from rlgym.rocket_league.reward_functions.touch_reward import TouchReward
# NEVER DO from file import *!
# This can lead to name conflicts that will be infinitely confusing down the line
# ALWAYS tell Python exactly what you want to import, makes sense for both you and Python.
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import RandomStateMutator


project_name="ExampleBot" #the name of your bot, changing this will start a new run 
            
from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import VariableTeamSizeMutator
        
from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import WeightedSampleMutator
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import RandomPhysicsMutator

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
    no_touch_timeout_seconds = 10 # how many seconds until we cut the episode short because the agent hasn't touched the ball
    game_timeout_seconds = 300 # full match length timeout.

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(
        (InAirReward(), 0.15), # Don't forget to jump!
        (SpeedTowardBallReward(), 5.0), # Move towards the ball!
        (VelocityBallToGoalReward(), 10.0), # Move the ball towards the goal!
        (TouchReward(), 50.0), # Big reward for touching the ball!
        (FaceBallReward(), 1.0), # Make sure we don't start driving backwards at the ball, too many times...
        (AdvancedTouchReward(touch_reward=0.5, acceleration_reward=1.0), 75.0), # Slightly more convoluted touch reward that also rewards powerful hits.
        (GoalReward(), 500.0) # I wouldn't set the goal scoring weight this high, but make sure scoring is the most important task.
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
        WeightedSampleMutator.from_zipped(
            (KickoffMutator(), 0.25),  #this means that 60% of the time, the ball and the cars will be in kickoff positions
            (RandomPhysicsMutator(), 0.75)   #this means that 40% of the time, the ball and the cars will be in random positions         
        ) # saucy rolv ratios :3
    )
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer() # THIS IS ONLY THE CLIENT! You will need to clone it in accordance with the readme.md in order to watch your bot play! 
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    # 32 processes
    n_proc = 32
    # Try increasing or decreasing this number as this has a direct correlation with performance.
    # I'd decrease it until performance started dropping to have maximum efficiency.

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    #Our code for loading our checkpoints
    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    checkpoint_files = os.listdir(checkpoint_folder)
    checkpoint_load_folder = os.path.join(checkpoint_folder, max(checkpoint_files)) if checkpoint_files else None

    # Basic hyperparameters, see what works for you!
    # Play around with them, don't take these for gospel
    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None, # Leave this empty, if you provide something here this is what will give you concrete game information from training, depending on what you add.
                      ppo_batch_size=100_000,  # batch size - much higher than 300K doesn't seem to help most people
                      policy_layer_sizes=[512, 512, 512],  # policy network layer sizes
                      critic_layer_sizes=[512, 512, 512],  # critic network layer sizes
                      ts_per_iteration=100_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=300_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,
                      render=True,
                      render_delay=0, # to create this you should define constants TICK_SKIP and TICK_RATE (120) and create a fraction that tells you how long one step is in one second.
                      add_unix_timestamp=False,
                      checkpoint_load_folder=checkpoint_load_folder,
                      checkpoints_save_folder=checkpoint_folder,                      # entropy coefficient - this determines the impact of exploration
                      policy_lr=2e-4, # policy learning rate, ensure this matches the critic learning rate.
                      device="auto", #device to use
                      critic_lr=2e-4,  # critic learning rate, keep the same as your policy's learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=10_000_000,  # save every 1M steps
                      timestep_limit=1e69,  # Train for an absurd number of steps, can be used if you have good hyperparameter staging or need to rapidly prototype things.
                      log_to_wandb=False # Set this to True if you want to use Weights & Biases for logging, Weights & Biases is generally optimal and the most used option.
                      ) 
    learner.learn() #we start training!
