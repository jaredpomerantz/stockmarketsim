# Stock Market Simulator

This repository contains the object definitions required to run stock market simulations with different policies. The objects in the `src/simulator/objects` directory are meant to simulate the progression of the market, including the acceptance of actions from participants, the resolution of orders, and the resulting effect on the stocks' states and participants' portfolios. 

The `src/simulators/objects/policies` directory can accept new policies. Currently, the BasePolicy class implements the Top-N Greedy Q-Learning policy mentioned in the case study associated with this simulator, located at the top level of this repository. New policy objects can inherit from this BasePolicy to implement new regression models and feature selections/augmentations. Future work should be done to abstract the order generation away from this Top-N Greedy Q-Learning policy to support entirely new policies, but this architecture was adequate for the completion of the case study mentioned above.

See `notebooks/run_market.ipynb` for an example of how to run the simulator, given that trained models are available.
