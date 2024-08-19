import gymnasium as gym
import numpy as np

# Define the total number of tests
TOTAL_TESTS = 10
equals = 0
for _ in range(TOTAL_TESTS):
    # Create the environment
    env = gym.make("ALE/JourneyEscape-v5")

    # Reset the environment to get the initial observation and any additional information
    observation, info = env.reset()

    # Clone the initial state
    initial_state = env.unwrapped.clone_state(include_rng=True)

    # Take an action (e.g., action 0)
    env.step(0)

    # Save the resulting state after the first action
    state_after_first_action = env.unwrapped.clone_state(include_rng=True)

    # Restore to the initial state
    env.unwrapped.restore_state(initial_state)

    # Take the same action again
    env.step(0)

    # Clone the state again after taking the same action
    state_after_restored_action = env.unwrapped.clone_state(include_rng=True)

    # Check if the two states are identical
    if np.array_equal(state_after_first_action, state_after_restored_action):
        equals += 1

# Print the result
print(f"Number of identical states: {equals} out of {TOTAL_TESTS}")
