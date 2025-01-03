from samples.option.OptionSolverMENTS import OptionSolverMENTS
from samples.option.USoptionMDP import USoptionMDP
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


mdp = USoptionMDP(option_type="Put", S0=S0, K=K, r=r, T=T, dt=dt, sigma=sigma)
ments_solver = OptionSolverMENTS(
    mdp,
    exploration_constant=1.0,
    discount_factor = 0.9,
    temperature = 1,
    epsilon = 0.1,
    verbose=False
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

fig, ax = plt.subplots(figsize=(10, 6))

# Process for root_rewards_1 (MENTS Vanilla)
mean_1 = root_rewards_1
std_1 = np.std(root_rewards_1)
ax.plot(iterations, mean_1, color='teal', label='MENTS Vanilla')
ax.fill_between(iterations, np.maximum(mean_1 - std_1, 0), mean_1 + std_1, color='teal', alpha=0.2)

# Process for root_rewards_2 (MENTS VC)
mean_2 = root_rewards_2
std_2 = np.std(root_rewards_2)
ax.plot(iterations, mean_2, color='coral', label='MENTS VC')
ax.fill_between(iterations, np.maximum(mean_2 - std_2, 0), mean_2 + std_2, color='coral', alpha=0.2)

# Labels and titles
ax.set_xlabel('MC Iteration')
ax.set_ylabel('Root Node State Value (log)')
ax.set_title('Value Convergence - Config E')
ax.legend()

# Add grid
plt.grid(True)
plt.show()