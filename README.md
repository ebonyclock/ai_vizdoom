AI learning from raw visual input using [ViZDoom](https://github.com/Marqt/ViZDoom) environment.

Everything is written in Python 2 (Theano + Lasagne) and is based on DeepMind's DQN.

Some videos with results:
https://www.youtube.com/watch?v=re6hkcTWVUY

---
To run some learning use learn.py. To change parameters change with setup variable and some initializers in agents.py. The learning process should add results in results directory and agent params in params directory. 

learn.py <PARAM_FILE> <RESULTS_FILE> <CONFIG_FILE> continues previously started and interrupted learning.
watch.py <PARAM_FILE> <CONFIG_FILE> runs 20 episodes with visualization.
plot_results.py <RESULTS_FILE> shows graph with test performance after each epoch.
