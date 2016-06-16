import argparse
import numpy as np


def build_learn_parser():
    parser = argparse.ArgumentParser(description='Learning script for ViZDoom.')

    agent_group = parser.add_mutually_exclusive_group(required=True)
    agent_group.add_argument('agent',
                             type=str, nargs="?", default=None,
                             help='agent function name from agents.py')
    agent_group.add_argument('--load-agent', '-l', metavar='<AGENT_FILE>', dest="agent_file", type=str, nargs=1,
                             default=[None],
                             help='load agent from a file')

    agent_group.add_argument('--list-agents', dest='list_agents', action='store_const',
                             const=True, default=False,
                             help='lists agents available in agents.py')

    agent_group.add_argument('--load-json', "-j", metavar='<JSON_FILE>', dest="json_file", type=str, nargs=1,
                             default=[None],
                             help="load agent's specification from a json file")

    parser.add_argument('--config-file', '-c', metavar='<CONFIG_FILE>', dest="config_file", type=str, nargs=1,
                        default=None,
                        help='configuration file (used only when loading agent or using json)')

    parser.add_argument('--name', '-n', metavar='<NAME>', type=str, nargs=1, default=[None],
                        help='agent\'s name (affects savefiles)')

    parser.add_argument('--no-save', dest='no_save', action='store_const',
                        const=True, default=False,
                        help='do not save agent\'s parameters')
    parser.add_argument('--no-save-results', dest='no_save_results', action='store_const',
                        const=True, default=False,
                        help='do not save agent\'s results')

    parser.add_argument('--no-save-best', dest='no_save_best', action='store_const',
                        const=True, default=False,
                        help='do not save the best agent')
    parser.add_argument('--epochs', '-e', metavar='<EPOCHS_NUM>', type=int, nargs=1, default=[np.inf],
                        help='number of epochs (default: infinity)')
    parser.add_argument('--train-steps', metavar='<TRAIN_STEPS>', type=int, nargs=1, default=[200000],
                        help='training steps per epoch (default: 200k)')
    parser.add_argument('--test-episodes', metavar='<TEST_EPISODES_NUM>', type=int, nargs=1, default=[300],
                        help='testing episodes per epoch (default: 300)')

    parser.add_argument('--no-tqdm', dest='no_tqdm', action='store_const',
                        const=True, default=False,
                        help='do not use tqdm progress bar')

    return parser
