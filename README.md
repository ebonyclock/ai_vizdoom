AI learning from raw visual input using [ViZDoom](https://github.com/Marqt/ViZDoom) environment.

Everything is written in Python 2 (Theano + Lasagne) and is based on DeepMind's DQN.

Some videos with results:
https://www.youtube.com/watch?v=re6hkcTWVUY

---
To launch learning use learn.py:
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
