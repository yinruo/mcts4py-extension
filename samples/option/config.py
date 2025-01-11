global_params = {
    "exploration_constant": 1.0,        
    "simulation_depth_limit": 100,   
    "discount_factor": 0.9, 
    "temperature": 0.7,
    "epsilon" : 0.2            
}

configurations = {
    "A": {"S0": 40, "K": 36, "T": 1, "r": 0.1, "sigma": 0.2, "dt": 0.1, "q": 0, "option_type": "call", **global_params},
    "B": {"S0": 12, "K": 10, "T": 1.5, "r": 0.08, "sigma": 0.25, "dt": 0.1, "q": 0.03, "option_type": "call", **global_params},
    "C": {"S0": 36, "K": 40, "T": 0.5, "r": 0.05, "sigma": 0.3, "dt": 0.05, "q": 0.05, "option_type": "put", **global_params},
    "D": {"S0": 10, "K": 12, "T": 1, "r": 0.12, "sigma": 0.35, "dt": 0.05, "q": 0.05, "option_type": "put", **global_params},
    "E": {"S0": 8, "K": 5, "T": 1.5, "r": 0.07, "sigma": 0.2, "dt": 0.1, "q": 0, "option_type": "call", **global_params},
    "F": {"S0": 5, "K": 8, "T": 1, "r": 0.1, "sigma": 0.4, "dt": 0.1, "q": 0.05, "option_type": "put", **global_params}
}


