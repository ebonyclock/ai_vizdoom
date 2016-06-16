AI learning from raw visual input using [ViZDoom](https://github.com/Marqt/ViZDoom) environment with [Theano](http://deeplearning.net/software/theano/) and [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html).

The code implements Double DQN with Duelling architecture:
- [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
- [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)  
- [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
- [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  


Some videos with early results (no double/duelling and bugs):
https://www.youtube.com/watch?v=re6hkcTWVUY

## Requirements:
- Python 2.7 (should work with 3 after correcting prints)
- [ViZDoom](https://github.com/Marqt/ViZDoom)
- [Scikit-image](http://scikit-image.org/)
- [Theano](http://deeplearning.net/software/theano/)
- [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html)
- [tqdm](https://github.com/tqdm/tqdm) (optional)

>>> The code requires vizdoom.so and vizdoom to be present in the root directory. Config files and scenarios are also needed (can be found in the [ViZDoom](https://github.com/openai/gym) repo).

## Usage of the learning script

```bash
usage: learn.py [-h] [--load-agent <AGENT_FILE>] [--list-agents]
                [--load-json <JSON_FILE>] [--config-file <CONFIG_FILE>]
                [--name <NAME>] [--no-save] [--no-save-results]
                [--no-save-best] [--epochs <EPOCHS_NUM>]
                [--train-steps <TRAIN_STEPS>]
                [--test-episodes <TEST_EPISODES_NUM>] [--no-tqdm]
                [agent]

Learning script for ViZDoom.

positional arguments:
  agent                 agent function name from agents.py

optional arguments:
  -h, --help            show this help message and exit
  --load-agent <AGENT_FILE>, -l <AGENT_FILE>
                        load agent from a file
  --list-agents         lists agents available in agents.py
  --load-json <JSON_FILE>, -j <JSON_FILE>
                        load agent's specification from a json file
  --config-file <CONFIG_FILE>, -c <CONFIG_FILE>
                        configuration file (used only when loading agent or
                        using json)
  --name <NAME>, -n <NAME>
                        agent's name (affects savefiles)
  --no-save             do not save agent's parameters
  --no-save-results     do not save agent's results
  --no-save-best        do not save the best agent
  --epochs <EPOCHS_NUM>, -e <EPOCHS_NUM>
                        number of epochs (default: infinity)
  --train-steps <TRAIN_STEPS>
                        training steps per epoch (default: 200k)
  --test-episodes <TEST_EPISODES_NUM>
                        testing episodes per epoch (default: 300)
  --no-tqdm             do not use tqdm progress bar

```

## Usage of the script for watching:
```bash
usage: watch.py [-h] [--config-file [config_file]] [--episodes [episodes]]
                [--no-watch] [--action-sleep [action_sleep]]
                [--episode-sleep [episode_sleep]]
                [agent_file]

A script to watch agents play or test them.

positional arguments:
  agent_file            file with the agent

optional arguments:
  -h, --help            show this help message and exit
  --config-file [config_file], -c [config_file]
                        override agent's configuration file
  --episodes [episodes], -e [episodes]
                        run this many episodes (default 20)
  --no-watch            do not display the window and do not sleep
  --action-sleep [action_sleep], -s [action_sleep]
                        sleep this many seconds after each action
                        (default=1/35.0)
  --episode-sleep [episode_sleep]
                        sleep this many seconds after each episode
                        (default=0.5)
```

