import mujoco
import mujoco.viewer
import numpy as np
import time

# =====================================================
# CONFIGURAÇÃO GERAL
# =====================================================
XML_PATH = "ur5e.xml"
T_TOTAL = 20.0          # duração total [s]
SEED = 0

np.random.seed(SEED)

# =====================================================
# CARREGAR MODELO
# =====================================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep

nq = model.nq
nv = model.nv
nu = model.nu

steps = int(T_TOTAL / dt)

# =====================================================
# TRAJETÓRIA EM ESPAÇO DE JUNTAS (SENO MULTIFREQUÊNCIA)
# =====================================================
q_center = np.zeros(nq)
q_amp = np.array([2.5, 2.0, 2.0, 3.0, 3.0, 3.0])
freqs = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
phases = np.random.uniform(0, 2*np.pi, nq)

def desired_trajectory(t):
    qd = q_center + q_amp * np.sin(2*np.pi*freqs*t + phases)
    dqd = 2*np.pi*freqs * q_amp * np.cos(2*np.pi*freqs*t + phases)
    ddqd = -(2*np.pi*freqs)**2 * q_amp * np.sin(2*np.pi*freqs*t + phases)
    return qd, dqd, ddqd

# =====================================================
# CONTROLE PD EM TORQUE
# =====================================================
Kp = np.diag([200, 200, 200, 50, 50, 50])
Kd = np.diag([20, 20, 20, 5, 5, 5])

# =====================================================
# BUFFERS DE LOG
# =====================================================
Q   = np.zeros((steps, nq))
dQ  = np.zeros((steps, nv))
ddQ = np.zeros((steps, nv))
Tau = np.zeros((steps, nu))

M_log = np.zeros((steps, nv, nv))
c_log = np.zeros((steps, nv))
g_log = np.zeros((steps, nv))

time_log = np.zeros(steps)

# =====================================================
# SIMULAÇÃO
# =====================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    for k in range(steps):
        t = k * dt

        # -------------------------
        # Trajetória desejada
        # -------------------------
        qd, dqd, ddqd = desired_trajectory(t)

        q = data.qpos.copy()
        dq = data.qvel.copy()

        # -------------------------
        # Torque PD
        # -------------------------
        tau = Kp @ (qd - q) + Kd @ (dqd - dq)
        data.ctrl[:] = tau

        # -------------------------
        # Passo de simulação
        # -------------------------
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        # =================================================
        # EXTRAÇÃO DINÂMICA
        # =================================================

        # -------- M(q)
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)

        # -------- g(q)  (chamada CORRETA do mj_rne)
        dq_backup = data.qvel.copy()
        data.qvel[:] = 0.0

        g = np.zeros(nv)
        mujoco.mj_rne(model, data, 0, g)

        data.qvel[:] = dq_backup

        # -------- c(q,dq)
        c = data.qfrc_bias - g

        # =================================================
        # LOG
        # =================================================
        Q[k]   = data.qpos.copy()
        dQ[k]  = data.qvel.copy()
        ddQ[k] = data.qacc.copy()
        Tau[k] = data.ctrl.copy()

        M_log[k] = M
        c_log[k] = c
        g_log[k] = g

        time_log[k] = t

# =====================================================
# SALVAR DATASET
# =====================================================
np.savez(
    "ur5e_full_dynamics_dataset.npz",
    time=time_log,
    q=Q,
    dq=dQ,
    ddq=ddQ,
    tau=Tau,
    M=M_log,
    c=c_log,
    g=g_log
)

print("✔ Dataset completo salvo em ur5e_full_dynamics_dataset.npz")
