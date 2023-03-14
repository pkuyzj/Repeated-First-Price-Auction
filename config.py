import argparse

arg_parser = argparse.ArgumentParser()

# ID of the data
arg_parser.add_argument('--data', type = str, default = '0')

# Random Seed
arg_parser.add_argument('--seed', type = int, default = 2023)

# Time Horizon
arg_parser.add_argument('--T', type = float, default = 1000000)

# Distribution of Private Values
arg_parser.add_argument('--distribution',
                        type = str, 
                        default = 'n',
                        choices = ['n', 'l', 'u'])
arg_parser.add_argument('--param1', type = float, default = 0.6)
arg_parser.add_argument('--param2', type = float, default = 0.1)
# Distribution of Competing Bids (or "Highest-Other-Bids")
arg_parser.add_argument('--mu', type = float, default= 0.4)
arg_parser.add_argument('--sigma', type = float, default= 0.1)

# Parameters for the algorithm
arg_parser.add_argument('--id', type = str, default = '0')
arg_parser.add_argument('--B', type = float, default = 10000)
arg_parser.add_argument('--epsilon', type = float, default = 0.001)
arg_parser.add_argument('--delta', type = float, default = 0.01)
arg_parser.add_argument('--M', type = int, default = 100)
arg_parser.add_argument('--K', type = int, default = 100)
arg_parser.add_argument('--alg',
                        type = str, 
                        default = 'PC',
                        choices = ['FC', 'FU', 'PC', 'PU'])

args = arg_parser.parse_args()