import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the saved file
df = pd.read_csv('samples/option/output/results_config_b.csv')

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

plt.title('Configuration B')
plt.xlabel('Planning Method')
plt.ylabel('Cum. Reward')
plt.show()
