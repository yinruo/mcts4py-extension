import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from samples.option.rewards import config_name
# Load the data from the saved file
df = pd.read_csv(f'samples/option/output/results_config_{config_name}.csv')

# Plot the data
plt.figure(figsize=(14, 8))
sns.set(style='whitegrid')

ax = sns.boxplot(
    data=df,
    notch=True,
    linewidth=2,
    fliersize=5,
    palette=['steelblue', 'lightgreen', 'steelblue', 'lightgreen', 'steelblue'],
    saturation=0.7
)

plt.title(f'Configuration {config_name}')
plt.xlabel('Planning Method')
plt.ylabel('Cum. Reward')
plt.show()
