HNP: Hyperspace Neighbour Penetration
======================================================================
Overview
--------
This repository provides model-free implementation of the [Hyperspace Neighbour Penetration (HNP)](https://arxiv.org/ftp/arxiv/papers/2106/2106.05497.pdf) reinforcement learning algorithm designed by TELUS **for environments with slowly changing continuous variables.**

Motivation
-------
Slowly changing continuous variables are an important part of many reinforcement learning problems. Though many algorithms have been proposed to handle these variables, they all have important issues:

- **Function approximation methods** which are capable of handling a continuous state space are very computationally intensive and lack interpretability, an important consideration when applying reinforcement learning to important real-world tasks
- **Tabular methods** which are more interpretable and less computationally intensive require discretization of the continous variables. 
If the discretization is too granular, the size of the state space explodes and the problem becomes computationally intractable. If the discretization is too coarse, the methods fail to differentiate between states during many consecutive timesteps due to the slowly-changing nature of the variables

HNP addresses these concerns by combining the **interpretability and computational efficiency of tabular methods** with the **ability of functional approximation methods to handle slowly-changing continuous variables**, thus making it the perfect choice to apply to a real-world problem with slowly-changing continuous variables.

Use cases of HNP
-----------------------------
HNP was initially designed for **data center HVAC control**, where the slowly-changing continuous variable is the air temperature. However, it could easily be applied to other problems such as:
* problem 1
* problem 2

Features
------


Documentation
------
- Getting Started with BeoBench
- Getting Started with Sinergym

License
-------

Indices and tables
==================

