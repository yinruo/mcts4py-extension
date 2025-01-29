import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from samples.atari.config import game_name

df = pd.read_csv(f'samples/atari/output/results_config_{game_name}.csv')

# Plot the data
plt.figure(figsize=(7, 7))
sns.set(style='whitegrid')

ax = sns.boxplot(
    data=df,
    notch=False,
    linewidth=2,
    fliersize=5,
    palette=['steelblue', 'lightgreen', 'steelblue'],
    saturation=0.7
)
plt.title(game_name)
plt.xlabel('Planning Method')
plt.ylabel('Cum. Reward')
plt.show()