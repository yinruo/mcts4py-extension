import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import samples.option.config as config

df = pd.read_csv(f'samples/option/output/results_config_{config.config_name}.csv')

# Plot the data
plt.figure(figsize=(7, 6))
sns.set(style='whitegrid')

ax = sns.boxplot(
    data=df,
    notch=True,
    linewidth=2,
    fliersize=5,
    palette=['steelblue', 'lightgreen', 'steelblue', 'lightgreen', 'steelblue'],
    saturation=0.7
)
plt.title(f'Configuration {config.config_name}')
plt.xlabel('Planning Method')
plt.ylabel('Cum. Reward')
plt.show()
