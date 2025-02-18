# MCTS library in Python

## Requirements

- Python 3.9+

## Samples

To run samples:
- run `pip install -e .` in the root directory
<!-- - run `python main.py` in the sample directory -->

To run gridworld:

- run `python -m samples.gridworld.main`in the root directory

To run CartPole game: 
- run `python -m samples.cartpole.main`in the root directory

To run Atari games: 
- run `gymnasium[atari]`
- run `pip install gymnasium[accept-rom-license]`
- adjust `game_name` in `config.py` to try out a specific Atari game.
- run `python -m samples.atari.rewards` to generate reward arrays for 3 different planning methods.
- run `python -m samples.atari.rewards_graph` to generate the comparing graph.

To run american option pricing : 
- run `python -m samples.option.rewards`to generate reward arrays for 5 different planning methods.
- run `python -m samples.option.rewards_graph` to generate the comparing graph.
- run `python -m samples.option.value_conv`to generate log reward arrays for root node for MENTS and MENTS VC
- run `python -m samples.option.value_conv_graph`to generate the above graph.
- run `python -m samples.option.plotting_v2` to generate the comparing graph for both rewards and root node rewards in .pynb file . 
- The generated reward comparison graphs can be found in `samples/option/output`.
- The generated reward comparison graphs for root node can be found in `samples/option/output_value_conv`.



```