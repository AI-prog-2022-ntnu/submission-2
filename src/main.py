from enviorments.hex.hex_game import HexGameEnvironment

env = HexGameEnvironment(4)
init_s = env.get_initial_state()

env.display_state(init_s)
print(env.is_state_won(init_s))