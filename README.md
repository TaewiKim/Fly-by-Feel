
# Fly-by-Feel


Implementations of RL algorithms used in the work of Fly-by-Feel. 
Note; you need a drone platform to run this code

## Usage
```bash
# e.g.
python train_sac.py 
python train_dqn.py 
```


## Description of Codes
- `environment.py` 
- `replayBuffer.py` -> replay buffer implementation
- `train_dqn.py` -> entry point for training 
- `train_sac.py` -> entry point for training 
- `models`
  - `dqn_model.py`  -> neural network implementations used for DQN algorithm
  - `sac_model.py`  -> neural network implementations used for SAC algorithm
- `utils`
  - `dwclient.py`  -> client code for communication between the drone and the algorithm server
  - `dwserver.py`  -> server code for communication between the drone and the algorithm server


## Dependencies
1. torch	2.0.1	
2. matplotlib	3.7.1
3. numpy	1.23.2
4. pandas	2.0.3
   

