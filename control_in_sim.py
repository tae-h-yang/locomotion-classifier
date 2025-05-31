import mujoco
import mujoco.viewer
import glfw
import numpy as np
import os

# Load model
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    this_dir, "descriptions", "toddlerbot", "toddlerbot_data_collection_scene.xml"
)
model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

model.opt.gravity[:] = 0

# Base motion parameters
MOVE_DISTANCE = 0.05  # meters
MOVE_DURATION = 0.05  # seconds
MOVE_STEPS = int(MOVE_DURATION / model.opt.timestep)
VELOCITY = MOVE_DISTANCE / MOVE_DURATION

KEY_MOVES = {
    glfw.KEY_UP: np.array([1, 0, 0]),
    glfw.KEY_DOWN: np.array([-1, 0, 0]),
    glfw.KEY_LEFT: np.array([0, 1, 0]),
    glfw.KEY_RIGHT: np.array([0, -1, 0]),
    glfw.KEY_E: np.array([0, 0, 1]),
    glfw.KEY_R: np.array([0, 0, -1]),
}

motion_queue = []

# Joint motion parameters
JOINT_MOVE_AMOUNT = 0.2  # radians
JOINT_MOVE_DURATION = 0.05  # seconds
JOINT_MOVE_STEPS = int(JOINT_MOVE_DURATION / model.opt.timestep)

# Get joint IDs
neck_yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw_drive")
neck_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch_act")

# Get joint qpos addresses
neck_yaw_qpos_adr = model.jnt_qposadr[neck_yaw_id]
neck_pitch_qpos_adr = model.jnt_qposadr[neck_pitch_id]

JOINT_KEYS = {
    glfw.KEY_B: (neck_yaw_qpos_adr, +1),  # turn head left
    glfw.KEY_C: (neck_yaw_qpos_adr, -1),  # turn head right
    glfw.KEY_F: (neck_pitch_qpos_adr, +1),  # head up
    glfw.KEY_V: (neck_pitch_qpos_adr, -1),  # head down
}

joint_motion_queue = []


def key_callback(key):
    if key in KEY_MOVES:
        motion_queue.append((KEY_MOVES[key], MOVE_STEPS))
    elif key in JOINT_KEYS:
        joint_qpos_idx, direction = JOINT_KEYS[key]
        joint_motion_queue.append(
            (joint_qpos_idx, direction * JOINT_MOVE_AMOUNT, JOINT_MOVE_STEPS)
        )


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print(
        "Arrow keys + E/R to move robot. F/C/V/B to control head joints. ESC to exit."
    )

    current_motion = None
    current_joint_motion = None

    while viewer.is_running():
        # Smooth base motion
        if current_motion is None and motion_queue:
            direction, steps = motion_queue.pop(0)
            current_motion = [direction, steps]

        if current_motion:
            direction, steps_left = current_motion
            step_move = direction * VELOCITY * model.opt.timestep
            data.qpos[:3] += step_move
            current_motion[1] -= 1
            if current_motion[1] <= 0:
                current_motion = None

        # Smooth joint motion
        if current_joint_motion is None and joint_motion_queue:
            joint_idx, total_delta, steps = joint_motion_queue.pop(0)
            delta_per_step = total_delta / steps
            current_joint_motion = [joint_idx, delta_per_step, steps]

        if current_joint_motion:
            joint_idx, delta_per_step, steps_left = current_joint_motion
            data.qpos[joint_idx] += delta_per_step
            current_joint_motion[2] -= 1
            if current_joint_motion[2] <= 0:
                current_joint_motion = None

        mujoco.mj_step(model, data)
        viewer.sync()
