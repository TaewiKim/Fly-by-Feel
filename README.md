
# Fly-by-Feel
## Important Note
The reinforcement learning experiments conducted in this study were performed using real-time data collected from flights in a real-world hardware environment, without relying on simulations.
As the code requires a hardware setup to function properly, there may be technical barriers to entry associated with hardware implementation.

The shared code below is provided to help you understand how the data is utilized in an actual hardware setup. We kindly ask for your understanding as you review the shared materials with this context in mind.

## Hardware
1. DEWE Soft Sirius (https://dewesoft.com/blog/sirius-uni-universal-amplifier)
2. Opti track PrimeX 13 (https://optitrack.com/cameras/primex-13/)
3. Opti track motion capture markers 6.4mm M3 Markers (https://optitrack.com/accessories/markers/)

## Usage
```bash
python train_sac.py 
```


## Description of Codes
- `environment.py` -> Initial setup Hardward environment 
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
   

