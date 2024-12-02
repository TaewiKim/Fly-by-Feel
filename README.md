
# Fly-by-Feel
## Important Note;
The reinforcement learning experiments conducted in this study were performed using real-time data collected from flights in a real-world hardware environment, without relying on simulations. As the code requires a hardware setup to function properly, we would like to note that there are limitations to directly reproducing the results. The shared code below is provided to help you understand how the data is utilized in an actual hardware setup. We kindly ask for your understanding as you review the shared materials with this context in mind.


## Usage
```bash
python train_sac.py 
```


## Description of Codes
- `environment.py` 
- `replayBuffer.py` -> replay buffer implementation
- `train_sac.py` -> entry point for training 
- `models`
  - `sac_model.py`  -> neural network implementations used for SAC algorithm
- `utils`
  - `dwclient.py`  -> client code for communication between the drone and the algorithm server
  - `dwserver.py`  -> server code for communication between the drone and the algorithm server


## Dependencies
1. torch	2.0.1	
2. matplotlib	3.7.1
3. numpy	1.23.2
4. pandas	2.0.3
   

