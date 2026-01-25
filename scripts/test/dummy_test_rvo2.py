import rvo2
import numpy as np

sim = rvo2.PyRVOSimulator(
    timeStep=0.1,
    neighborDist=5.0,
    maxNeighbors=10,
    timeHorizon=5.0,
    timeHorizonObst=5.0,
    radius=0.3,
    maxSpeed=1.0
)

# Add two robots
id1 = sim.addAgent((0.5, 0.5))
id2 = sim.addAgent((4.5, 0.5))

goal1 = np.array([4.5, 4.5])
goal2 = np.array([0.5, 4.5])

for _ in range(200):
    for i, goal in zip([id1, id2], [goal1, goal2]):
        pos = np.array(sim.getAgentPosition(i))
        v_pref = goal - pos
        if np.linalg.norm(v_pref) > 1e-3:
            v_pref = v_pref / np.linalg.norm(v_pref)
        sim.setAgentPrefVelocity(i, tuple(v_pref))

    sim.doStep()

    print(sim.getAgentPosition(id1), sim.getAgentPosition(id2))
