from mcts4py.TestOptionMCTS import TestOptionMCTS
from samples.option.TestUSoptionMDP import TestUSoptionMDP

S0_1 = 1
K_1=0.9
T_1=1
r_1=0
sigma_1=0.15
div_yield_1=0

mdp = TestUSoptionMDP(option_type="Put", S0=S0_1, K=K_1, r=r_1, T=T_1, dt=1/10, sigma=sigma_1)
US_solver = TestOptionMCTS(
    mdp,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=True
)

US_solver.run_option()

""" from mcts4py.ExpectationSolverOptionMCTS import ExpectationSolverOption
from samples.option.USoptionMDPOG import USoptionMDPOG

S0_1 = 1
K_1=0.9
T_1=1
r_1=0
sigma_1=0.15
div_yield_1=0


mdp_1 = USoptionMDPOG(option_type="Put", S0=S0_1, K=K_1, r=r_1, T=T_1, dt=0.1, sigma=sigma_1)
expect_US_solver_1 = ExpectationSolverOption(
    mdp_1,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
expect_mcts_price_1 = expect_US_solver_1.run_option()

print("expect_mcts_price",expect_mcts_price_1 ) """