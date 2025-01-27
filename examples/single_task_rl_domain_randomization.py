from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from rlbench import ArmActionMode
from rlbench import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.tasks import DislPickUpBlueCup
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all(True)

# We will borrow some from the tests dir
rand_config = VisualRandomizationConfig(
    image_directory='../tests/unit/assets/textures')

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = DomainRandomizationEnvironment(
    action_mode, obs_config=obs_config, headless=False,
    randomize_every=RandomizeEvery.EPISODE, frequency=1,
    visual_randomization_config=rand_config
)
env.launch()

task = env.get_task(DislPickUpBlueCup)

agent = Agent(env.action_size)

training_steps = 5000
episode_length = 5
total_eps = training_steps // episode_length
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print(f'Reset Episode {i//episode_length + 1} of {total_eps}')
        try:
            descriptions, obs = task.reset()
        except RuntimeError as e:
            print(f'RUNTIME ERROR: {e}')
            i -= 1  # reset
            next

    action = agent.act(obs)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
