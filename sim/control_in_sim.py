import os
import numpy as np
import mujoco
import mujoco.viewer
import glfw
from scipy.spatial.transform import Rotation as R


class ToddlerbotSimulator:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = 0

        # Base motion parameters
        self.MOVE_DISTANCE = 0.1
        self.MOVE_DURATION = 0.05
        self.MOVE_STEPS = int(self.MOVE_DURATION / self.model.opt.timestep)
        self.VELOCITY = self.MOVE_DISTANCE / self.MOVE_DURATION
        self.motion_queue = []

        # Joint motion parameters
        self.JOINT_MOVE_AMOUNT = 0.4
        self.JOINT_MOVE_DURATION = 0.05
        self.JOINT_MOVE_STEPS = int(self.JOINT_MOVE_DURATION / self.model.opt.timestep)
        self.joint_motion_queue = []

        # Key mappings: updated LEFT/RIGHT to rotate body instead of translate
        self.KEY_MOVES = {
            glfw.KEY_UP: np.array([1, 0, 0]),  # Forward
            glfw.KEY_DOWN: np.array([-1, 0, 0]),  # Backward
            glfw.KEY_LEFT: "turn_left",  # Rotate left
            glfw.KEY_RIGHT: "turn_right",  # Rotate right
            glfw.KEY_E: np.array([0, 0, 1]),  # Up
            glfw.KEY_R: np.array([0, 0, -1]),  # Down
        }

        # Rotation angle per turn (in radians)
        self.ROTATION_ANGLE = np.deg2rad(10)  # 10 degree turns

        self._init_joint_indices()

    def _init_joint_indices(self):
        yaw_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw_drive"
        )
        pitch_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch_act"
        )
        yaw_qpos_adr = self.model.jnt_qposadr[yaw_id]
        pitch_qpos_adr = self.model.jnt_qposadr[pitch_id]

        self.JOINT_KEYS = {
            glfw.KEY_B: (yaw_qpos_adr, +1),
            glfw.KEY_C: (yaw_qpos_adr, -1),
            glfw.KEY_F: (pitch_qpos_adr, +1),
            glfw.KEY_V: (pitch_qpos_adr, -1),
        }

    def key_callback(self, key):
        if key in self.KEY_MOVES:
            move = self.KEY_MOVES[key]
            if isinstance(move, str):
                # Extract current orientation as quaternion
                quat = self.data.qpos[3:7]
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # wxyz → xyzw

                # Apply Z-axis rotation
                if move == "turn_left":
                    delta_rot = R.from_euler("z", self.ROTATION_ANGLE)
                elif move == "turn_right":
                    delta_rot = R.from_euler("z", -self.ROTATION_ANGLE)

                new_rot = delta_rot * rot
                new_quat = new_rot.as_quat()  # xyzw

                # Store back in wxyz format
                self.data.qpos[3:7] = [
                    new_quat[3],
                    new_quat[0],
                    new_quat[1],
                    new_quat[2],
                ]
            else:
                self.motion_queue.append((move, self.MOVE_STEPS))
        elif key in self.JOINT_KEYS:
            joint_idx, direction = self.JOINT_KEYS[key]
            delta = direction * self.JOINT_MOVE_AMOUNT
            self.joint_motion_queue.append((joint_idx, delta, self.JOINT_MOVE_STEPS))

    def _apply_base_motion(self, current_motion):
        direction, steps_left = current_motion

        # Get current orientation (quat wxyz → scipy xyzw)
        quat = self.data.qpos[3:7]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

        # Rotate the direction vector into world frame
        world_direction = rot.apply(direction)

        # Compute and apply world-frame movement
        step_move = world_direction * self.VELOCITY * self.model.opt.timestep
        self.data.qpos[:3] += step_move

        return (direction, steps_left - 1) if steps_left > 1 else None

    def _apply_joint_motion(self, joint_motion):
        joint_idx, delta_per_step, steps_left = joint_motion
        self.data.qpos[joint_idx] += delta_per_step
        return (joint_idx, delta_per_step, steps_left - 1) if steps_left > 1 else None

    def run(self):
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as viewer:
            print(
                "Arrow keys + E/R to move robot. F/C/V/B to control head joints. ESC to exit."
            )
            current_motion = None
            current_joint_motion = None

            while viewer.is_running():
                if current_motion is None and self.motion_queue:
                    current_motion = self.motion_queue.pop(0)
                if current_motion:
                    current_motion = self._apply_base_motion(current_motion)

                if current_joint_motion is None and self.joint_motion_queue:
                    joint_idx, total_delta, steps = self.joint_motion_queue.pop(0)
                    delta_per_step = total_delta / steps
                    current_joint_motion = (joint_idx, delta_per_step, steps)
                if current_joint_motion:
                    current_joint_motion = self._apply_joint_motion(
                        current_joint_motion
                    )

                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(
        this_dir, "../descriptions/toddlerbot/toddlerbot_data_collection_scene_01.xml"
    )
    sim = ToddlerbotSimulator(xml_path)
    sim.run()
