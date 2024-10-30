from samples.option.ls_american.option import Option 
from samples.option.ls_american.process import HestonProcess
from samples.option.ls_american.pricing import monte_carlo_simulation, monte_carlo_simulation_LS

put_1 = Option(s0=1, v0=0.15, T=1, K=0.95, call=False)
heston = HestonProcess(mu=0.06, kappa=0.0005, theta=0.04, eta=0.1, rho=-0.5)
monte_carlo_simulation_LS(option=put_1, process=heston, n=25_000, m=252)