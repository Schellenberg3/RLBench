from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

#############################################################################
# Custom task for the DISL group at Ohio State.                             #
#                                                                           #
# NOTE that this overrides that get_low_dim_state of the parent class and   #
# allows us to control what information about the position of the gripper   #
# and cup is saved at each time step.                                       #
#############################################################################


# Note that the class name MUST be a camel-case version of the file name
class DislPickUpBlueCup(Task):

    def init_task(self) -> None:
        self.cup = Shape('cup')
        self.cup_visual = Shape('cup_visual')
        self.boundary = Shape('boundary')
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.cup])
        self.register_success_conditions([
            DetectedCondition(self.cup, self.success_sensor, negated=True),
            GraspedCondition(self.robot.gripper, self.cup),
        ])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = 0
        target_color_name, target_rgb = colors[4]  # gets ('blue', s(0.0,0.0,1.0))

        self.cup_visual.set_color(target_rgb)

        b = SpawnBoundary([self.boundary])
        b.clear()

        # prevents the cup (defined from the sensor) from rotating
        b.sample(self.success_sensor, 
                 min_distance=0.1,
                 min_rotation=(0, 0, 0),
                 max_rotation=(0, 0, 0))

        return ['pick up the %s cup' % target_color_name,
                'grasp the %s cup and lift it' % target_color_name,
                'lift the %s cup' % target_color_name]

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        # Prevents the base itself from rotating
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def reward(self): # -> Union[float, None]:
        """Allows the user to customise the task and add reward shaping."""
        return None

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        """ Custom low dimensional state for this task that returns the cup then gripper pose as
        an array (X,Y,Z,Qx,Qy,Qz,Qw) for both objects.
        """
        cup_pose = self.cup.get_pose().copy()
        gripper_pose = self.robot.gripper.get_pose().copy()

        return np.array([cup_pose,
                         gripper_pose])