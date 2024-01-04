
# Fly-by-Feel


Implementations of RL algorithms used in the work of Fly-by-Feel

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
1. PyTorch
2. OpenAI GYM 

