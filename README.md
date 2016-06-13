AI learning from raw visual input using [ViZDoom](https://github.com/Marqt/ViZDoom) environment with [Theano](http://deeplearning.net/software/theano/) and [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html).

The code implements Double DQN with Duelling architecture:
[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  


Some videos with results:
https://www.youtube.com/watch?v=re6hkcTWVUY

## Requirements:
- Python 2.7 (should work with 3 after correcting prints)
- [ViZDoom](https://github.com/openai/gym)
- [Scikit-image](http://scikit-image.org/)
- [Theano](http://deeplearning.net/software/theano/)
- [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html)
- [tqdm](https://github.com/tqdm/tqdm) (optional)

>>> Code requires vizdoom.so and vizdoom to be present in the root directory. Config files and scenarios are also needed (can be found in the [ViZDoom](https://github.com/openai/gym) repo).

## Usage of learn.py

```bash
usage: learn.py [-h] [--load-agent [agent_file]] [--config-file [config_file]]
                [--name [name]] [--no-save] [--no-save-results]
                [--no-save-best] [--epochs [epochs]]
                [--train-steps [train_steps]]
                [--test-episodes [test_episodes]] [--no-tqdm]
                [agent]

Learning script for ViZDoom.

positional arguments:
  agent                 agent function name from agents.py

optional arguments:
  -h, --help            show this help message and exit
  --load-agent [agent_file], -l [agent_file]
                        load agent from a file
  --config-file [config_file], -c [config_file]
                        configuration file (used only when loading agent
  --name [name], -n [name]
                        agent's name (affects savefiles)
  --no-save             do not save agent's parameters
  --no-save-results     do not save agent's results
  --no-save-best        do not save the best agent
  --epochs [epochs], -e [epochs]
                        number of epochs (default infinity)
  --train-steps [train_steps]
                        training steps per epoch (default 200k)
  --test-episodes [test_episodes]
                        testing episodes per epoch (default 300)
  --no-tqdm             do not use tqdm progress bar

```
