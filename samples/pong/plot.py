
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
results_dict = {
    "Vanilla MCTS": [2490,1430,2230,1800,2560],
    "1/2-greedy + UCT MCTS": [2020,2450,1450,1730,2930],
    "MENTS": [3100,1510,1830,2310,3290],
}


df = pd.DataFrame(results_dict)

# Plot using seaborn
plt.figure(figsize=(14, 8))
sns.boxplot(data=df)
plt.title("Riverraid Reward Estimates Using Different Methods")
plt.xlabel("Algorithm")
plt.ylabel("Reward Estimate")
plt.show()