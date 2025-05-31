import mujoco
import mujoco.viewer
import glfw
import numpy as np
import os
import time

# Load model
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    this_dir, "descriptions", "toddlerbot", "toddlerbot_data_collection_scene.xml"
)
model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

model.opt.gravity[:] = 0

# Move 0.3 meters over 0.5 seconds
MOVE_DISTANCE = 0.1
MOVE_DURATION = 0.1  # seconds
MOVE_STEPS = int(MOVE_DURATION / model.opt.timestep)
VELOCITY = MOVE_DISTANCE / MOVE_DURATION  # meters/sec

KEY_MOVES = {
    glfw.KEY_UP: np.array([1, 0, 0]),
    glfw.KEY_DOWN: np.array([-1, 0, 0]),
    glfw.KEY_LEFT: np.array([0, 1, 0]),
    glfw.KEY_RIGHT: np.array([0, -1, 0]),
    glfw.KEY_E: np.array([0, 0, 1]),
    glfw.KEY_R: np.array([0, 0, -1]),
}

# Motion queue: (direction, steps remaining)
motion_queue = []


def key_callback(key):
    if key in KEY_MOVES:
        motion_queue.append((KEY_MOVES[key], MOVE_STEPS))


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("Press arrow/Q/E keys to move robot smoothly. ESC to exit.")

    current_motion = None

    while viewer.is_running():
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

        mujoco.mj_step(model, data)
        viewer.sync()
