Getting Started with BeoBench
===============

BeoBench is a toolkit to conduct experiments on using reinforcement learning for HVAC control. Below you will find instructions for a minimalist example of training HNP on BeoBench


Requirements
------------

* Python version >= 3.8

HNP Installation
-----
From PyPI:

```pip install hnp```

From source:

```
git clone https://github.com/VectorInstitute/MBRL-HVAC-Energy-Optimization/tree/thomas-dev
cd MBRL-HVAC-Energy-Optimization/dev_env/src/
pip install .
```

BeoBench Installation
---------
Follow the instructions [here](https://beobench.readthedocs.io/en/latest/getting_started.html)


Example Usage with BeoBench
-------------
Put the following in a file called `config.yaml`:
.. code-block:: yaml

    agent:
        # script to run inside experiment container
        origin: ./train_hnp.py
        # configuration that can be accessed by script above
        config:
            num_episodes: 50
    env:
    # gym framework from which we want use an environment
    gym: sinergym
    # gym-specific environment configuration
    config:
        # sinergym environment name
        name: Eplus-5Zone-hot-discrete-v1
        # whether to normalise observations
        normalize: True
    wrappers: [] # no wrappers added for this example
    general:
    # save experiment data to ``./beobench_results`` directory
    local_dir: ./beobench_results

    hyperparams:
    eps_annealing: 0.999
    lr_annealing: 0.999
    horizon: 10000

Put the following in a file called `train_hnp.py` in the same directory as the `config.yaml` file:

.. code-block:: python

    from beobench.experiment.provider import create_env, config

    from hnp.agent_hnp import HNPAgent, ObservationWrapper
    import numpy as np

    # Create environment and wrap observations
    obs_to_keep = np.array([0, 8]) 
    lows = np.array([0,0])
    highs = np.array([1,1]) # if discrete then is number of actions
    mask = np.array([0, 0]) 
    env = create_env()

    env = ObservationWrapper(env, obs_to_keep, lows, highs, mask)
    agent = HNPAgent(
        env, 
        mask,
        lows,
        highs)

    agent.learn(config["agent"]["config"]["num_episodes"])
    agent.save_results()
    env.close()

Run `beobench run --config config.yaml`



















