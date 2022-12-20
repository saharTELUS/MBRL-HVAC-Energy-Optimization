Getting Started 
===============

Integrate HNP into your project quickly!


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

Example Usage with BeoBench
-------------
BeoBench is a toolkit to conduct experiments on using reinforcement learning for HVAC control
This is a minimalist example of HNP being used to optimize data center HVAC control using the BeoBench HVAC simulator.
First install beobench 


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

















