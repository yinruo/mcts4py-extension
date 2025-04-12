import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import samples.option.config as config

df = pd.read_csv(f'samples/option/output/v_4/results_multi_{config.multi_option_config_name}.csv')

# Plot the data
plt.figure(figsize=(7, 7))
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
