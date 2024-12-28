
from mcts4py.ments.SolverMents import StatefulSolverMENTS
from mcts4py.option.newSolverMENTS import NewOptionMENTS
from samples.option.TestUSoptionMDP import TestUSoptionMDP
import matplotlib.pyplot as plt
import numpy as np
S0 = 10
K = 12
T = 2
r = 0
sigma = 0.4
dt = 1/20
div_yield = 0
# Configuration for simulations
simulation_depth_limit = 100
exploration_constant = 1.0
num_runs = 200


mdp = TestUSoptionMDP(option_type="Put", S0=S0, K=K, r=r, T=T, dt=dt, sigma=sigma)
ments_solver = StatefulSolverMENTS(
    mdp,
    simulation_depth_limit = 1000,
    discount_factor = 0.6,
    exploration_constant=1.0
    
)

""" ments_solver = NewOptionMENTS(
    mdp,
    exploration_constant=1.0,
    discount_factor = 0.9,
    temperature = 1,
    epsilon = 0.1,
    verbose=False
) """

root_rewards_1 = ments_solver.get_root_rewards()
root_rewards_2 = ments_solver.get_root_rewards_hindsight()


iterations = np.arange(len(root_rewards_1)) 

fig, ax = plt.subplots(figsize=(10,6))

mean_1 = root_rewards_1
std_1 = np.std(root_rewards_1) 
ax.plot(iterations, mean_1, color='teal', label='MENTS Vanilla')
ax.fill_between(iterations, mean_1-std_1, mean_1+std_1, color='teal', alpha=0.2)

mean_2 = root_rewards_2
std_2 = np.std(root_rewards_2)
ax.plot(iterations, mean_2, color='coral', label='MENTS VC')
ax.fill_between(iterations, mean_2-std_2, mean_2+std_2, color='coral', alpha=0.2)

ax.set_xlabel('MC Iteration')
ax.set_ylabel('Root Node State Value (log)')
ax.set_title('Value Convergence - Config E')
ax.legend()

plt.grid(True)
plt.show()
