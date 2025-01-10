from samples.option.OptionSolverMENTS import OptionSolverMENTS
from samples.option.USoptionMDP import USoptionMDP
import matplotlib.pyplot as plt
import numpy as np
import samples.option.config as config
import pandas as pd



mdp = USoptionMDP(option_type=config.option_type, S0=config.S0, K=config.K, r=config.r, T=config.T, dt=config.dt, sigma=config.sigma, q=config.q)
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

root_rewards_1 = np.array(root_rewards_1)
root_rewards_2 = np.array(root_rewards_2)

root_rewards_1 = np.log(root_rewards_1 )
root_rewards_2 = np.log(root_rewards_2 )


# Save results to a CSV file
df = pd.DataFrame({
    "iterations": np.arange(len(root_rewards_1)),
    "root_rewards_1": root_rewards_1,
    "root_rewards_2": root_rewards_2
})

# Save to CSV
output_file = "samples/option/output_value_conv/value_convergence_config_b.csv"
df.to_csv(output_file, index=False)
print(f"Root rewards saved to {output_file}")
