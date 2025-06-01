import os
import cv2
import numpy as np
from datetime import datetime
import mujoco
import mujoco.viewer
import glfw
from sim.control_in_sim import ToddlerbotSimulator

# Output setup
RES_X, RES_Y = 640, 480
os.environ["MUJOCO_GL"] = "glfw"
this_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(this_path, "../data/videos")
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%m%d%H%M")  # MonthDateHourMinute

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
rgb_writer = cv2.VideoWriter(
    os.path.join(output_dir, f"rgb_video_{timestamp}.mp4"),
    fourcc,
    30.0,
    (RES_X, RES_Y),
)
depth_writer = cv2.VideoWriter(
    os.path.join(output_dir, f"depth_video_{timestamp}.mp4"),
    fourcc,
    30.0,
    (RES_X, RES_Y),
    isColor=False,
)


def render_and_save(renderer, data):
    # RGB from left camera
    renderer.update_scene(data, camera="head_cam_left")
    rgb = renderer.render()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Depth from left camera
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="head_cam_left")
    depth = renderer.render()
    renderer.disable_depth_rendering()

    max_depth = 3.0
    depth_vis = (np.clip(depth, 0, max_depth) / max_depth * 255).astype(np.uint8)

    rgb_writer.write(rgb)
    depth_writer.write(depth_vis)


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(
        this_dir, "../descriptions/toddlerbot/toddlerbot_data_collection_scene_01.xml"
    )
    sim = ToddlerbotSimulator(xml_path)

    # Set up renderer
    gl_ctx = mujoco.GLContext(RES_X, RES_Y)
    gl_ctx.make_current()
    renderer = mujoco.Renderer(sim.model, width=RES_X, height=RES_Y)

    SAVE_INTERVAL = 0.01  # Save frame every 0.2 seconds of simulation time
    last_save_time = 0.0

    print("Control the robot using keyboard. Video will be saved to ../data/videos/")
    with mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=sim.key_callback
    ) as viewer:
        current_motion = None
        current_joint_motion = None

        while viewer.is_running():
            if current_motion is None and sim.motion_queue:
                current_motion = sim.motion_queue.pop(0)
            if current_motion:
                current_motion = sim._apply_base_motion(current_motion)

            if current_joint_motion is None and sim.joint_motion_queue:
                joint_idx, total_delta, steps = sim.joint_motion_queue.pop(0)
                delta_per_step = total_delta / steps
                current_joint_motion = (joint_idx, delta_per_step, steps)
            if current_joint_motion:
                current_joint_motion = sim._apply_joint_motion(current_joint_motion)

            mujoco.mj_step(sim.model, sim.data)
            viewer.sync()

            current_time = sim.data.time
            if current_time - last_save_time >= SAVE_INTERVAL:
                render_and_save(renderer, sim.data)
                last_save_time = current_time

            # render_and_save(renderer, sim.data)

    rgb_writer.release()
    depth_writer.release()
    print("Recording complete. Files saved to ../data/videos/")


if __name__ == "__main__":
    main()
