import mujoco
import mujoco_viewer

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

viewer = mujoco_viewer.MujocoViewer(m, d)

while viewer.is_alive:
    mujoco.mj_step(m, d)
    viewer.render()

viewer.close()
