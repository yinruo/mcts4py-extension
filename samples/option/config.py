config_name = "E"
multi_option_config_name = "C"
global_params = {
    "exploration_constant": 1.0,        
    "simulation_depth_limit": 100,   
    "discount_factor": 0.6, 
    "temperature": 0.7,
    "epsilon" : 0.2
}

configurations = {
    "A": {"S0": 40, "K": 36, "T": 1, "r": 0.1, "sigma": 0.2, "dt": 0.1, "q": 0, "option_type": "call", **global_params},
    "B": {"S0": 12, "K": 10, "T": 2, "r": 0.08, "sigma": 0.25, "dt": 0.1, "q": 0.03, "option_type": "call", **global_params},
    "C": {"S0": 36, "K": 40, "T": 1, "r": 0.05, "sigma": 0.3, "dt": 0.02, "q": 0.05, "option_type": "put", **global_params},
    "D": {"S0": 10, "K": 14, "T": 1, "r": 0.12, "sigma": 0.35, "dt": 0.1, "q": 0.05, "option_type": "put", **global_params},
    "E": {"S0": 8, "K": 5, "T": 1.5, "r": 0.07, "sigma": 0.2, "dt": 0.1, "q": 0, "option_type": "call", **global_params},
    "F": {"S0": 5, "K": 8, "T": 1, "r": 0.1, "sigma": 0.4, "dt": 0.01, "q": 0.05, "option_type": "put", **global_params}
}

multi_option_configurations = {
    "A": {
        "S0_list": [22, 35, 18, 12, 28],
        "K_list":  [20, 30, 20, 10, 25],
        "T": 1.5,
        "r": 0.07,
        "sigma_list": [0.25, 0.3, 0.35, 0.4, 0.2],
        "q_list": [0.02, 0.03, 0.01, 0.05, 0.02],
        "dt": 0.05,
        "option_type_list": ["put", "call", "put", "put", "call"],
        "max_exercise_per_step": 3,
        **global_params
    },

    "B": {
        "S0_list": [40, 12, 8],
        "K_list": [36, 10, 5],
        "T": 1.0,
        "r": 0.08,
        "sigma_list": [0.2, 0.25, 0.2],
        "q_list": [0, 0.03, 0],
        "dt": 0.02,
        "option_type_list": ["call", "call", "call"],
        "max_exercise_per_step": 3,
        **global_params
    },

    "C": {
        "S0_list": [25, 30, 15, 20],
        "K_list": [20, 28, 16, 18],
        "T": 1.5,
        "r": 0.06,
        "sigma_list": [0.3, 0.35, 0.25, 0.4],
        "q_list": [0.02, 0.01, 0.04, 0.03],
        "dt": 0.05,
        "option_type_list": ["call", "put", "call", "put"],
        "max_exercise_per_step": 3,
        **global_params
    },
"D": {
    "S0_list": [26, 15, 38],
    "K_list":  [24, 16, 35],
    "T": 1.3,
    "r": 0.06,
    "sigma_list": [0.27, 0.3, 0.25],
    "q_list": [0.02, 0.03, 0.015],
    "dt": 0.025,
    "option_type_list": ["call", "put", "call"],
    "max_exercise_per_step": 2,
    **global_params
},
"E": {
    "S0_list": [40, 35, 50, 45, 60],
    "K_list":  [42, 36, 48, 47, 58],
    "T": 1.5,
    "r": 0.05,
    "sigma_list": [0.25, 0.3, 0.28, 0.26, 0.27],
    "q_list": [0.01, 0.015, 0.02, 0.01, 0.015],
    "dt": 0.025,
    "option_type_list": ["call", "put", "call", "put", "call"],
    "max_exercise_per_step": 3,
    **global_params
},
"K": {
    "S0_list": [100, 95, 110, 105, 90],
    "K_list":  [98, 94, 112, 102, 91],
    "T": 0.8,
    "r": 0.07,
    "sigma_list": [0.3, 0.32, 0.28, 0.31, 0.29],
    "q_list": [0.01, 0.015, 0.01, 0.02, 0.01],
    "dt": 0.02,
    "option_type_list": ["call", "call", "call", "call", "put"],
    "max_exercise_per_step": 4,
    **global_params
},




}



