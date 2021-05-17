from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import DislPickUpBlueCup
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##################################################################################################
# This script is for testing methods to record the pose information for the cup and gripper      #
# in the disl_pick_up_blue_cup.py task. It runs one demo before displaying the saved information #
##################################################################################################

live_demos = True
DATASET = ''

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

task = env.get_task(DislPickUpBlueCup)

NUM_DEMOS = 1  # Do not change for this example, only one demo returns -> List[Observation]
demos = task.get_demos(NUM_DEMOS, live_demos=live_demos)  # More than one demo returns -> List[List[Observation]]
demos = np.array(demos).flatten()

print('POSITIONS')
steps = np.shape(demos)[0]

verbose = False
if verbose:
    [print(f'step {i} of steps: cup at ', d.task_low_dim_state[0][:3],
           f'\n              gripper at ', d.task_low_dim_state[1][:3], '\n') for i, d in enumerate(demos)]

cup = []
gripper = []
[cup.append(d.task_low_dim_state[0]) for d in demos]
[gripper.append(d.task_low_dim_state[1]) for d in demos]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cup = np.array(cup)
gripper = np.array(gripper)

ax.plot(cup[:, 0], cup[:, 1], cup[:, 2])
ax.plot(gripper[:, 0], gripper[:, 1], gripper[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 0.5)
ax.set_zlim(0.7, 2)

plt.show()

print('Done')
env.shutdown()
