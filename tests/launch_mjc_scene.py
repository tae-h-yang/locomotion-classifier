import os
import mujoco
import mujoco.viewer

# Load the model with the XML that includes your <camera name="head_cam" ... />
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    this_dir, "descriptions", "toddlerbot", "toddlerbot_table_scene.xml"
)
model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

# Optional: no gravity, for static visualization
model.opt.gravity[:] = 0

# Launch viewer (interactive)
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
