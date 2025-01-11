import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import samples.option.config as config  # Import configuration file
from samples.option.value_conv import config_name
# Load the data from the CSV file
input_file = f"samples/option/output_value_conv/value_convergence_config_{config_name}.csv"
df = pd.read_csv(input_file)

# Extract data for plotting
iterations = df["iterations"]
root_rewards_1 = df["root_rewards_1"]
root_rewards_2 = df["root_rewards_2"]

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

# Plot root_rewards_1 (MENTS Vanilla)
mean_1 = root_rewards_1
std_1 = np.std(root_rewards_1)
ax.plot(iterations, mean_1, color='teal', label='MENTS Vanilla')
ax.fill_between(iterations, np.maximum(mean_1 - std_1, 0), mean_1 + std_1, color='teal', alpha=0.2)

# Plot root_rewards_2 (MENTS VC)
mean_2 = root_rewards_2
std_2 = np.std(root_rewards_2)
ax.plot(iterations, mean_2, color='coral', label='MENTS VC')
ax.fill_between(iterations, np.maximum(mean_2 - std_2, 0), mean_2 + std_2, color='coral', alpha=0.2)

# Labels and titles
ax.set_xlabel(config.x_label if hasattr(config, "x_label") else "MC Iteration")
ax.set_ylabel(config.y_label if hasattr(config, "y_label") else "Root Node State Value (log)")
ax.set_title(config.plot_title if hasattr(config, "plot_title") else f"Value Convergence - Config {config_name}")
ax.legend()

# Add grid and show plot
plt.grid(True)
plt.show()
