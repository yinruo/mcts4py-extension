import pandas as pd
from scipy.stats import ttest_ind
import samples.option.config as config
df = pd.read_csv(f'samples/option/output/v_2/results_config_{config.config_name}.csv')

""" first = df["UCT"]
second = df["UCT VC"] """

first = df["LS"]
second = df["MENTS"]
t_stat, p_value = ttest_ind(first, second, equal_var=False)  # Welch’s t-test 更稳健

print(f"t 值: {t_stat:.4f}, p 值: {p_value:.10f}")
