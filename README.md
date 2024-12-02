
# Fly-by-Feel
Implementations of RL algorithms used in the work of 'Wing-strain-based flight control of flapping-wing drones through reinforcement learning'
(https://doi.org/10.1038/s42256-024-00893-9)

## Important Note
The reinforcement learning experiments conducted in this study were performed using real-time data collected from flights in a real-world hardware environment, without relying on simulations.
As the code requires a hardware setup to function properly, there may be technical barriers to entry associated with hardware implementation.

The shared code below is provided to help you understand how the data is utilized in an actual hardware setup. We kindly ask for your understanding as you review the shared materials with this context in mind.

## Hardware
1. DEWE Soft Sirius (https://dewesoft.com/blog/sirius-uni-universal-amplifier)
2. Opti track PrimeX 13 (https://optitrack.com/cameras/primex-13/)
3. Opti track motion capture markers (https://optitrack.com/accessories/markers/)
4. Teensyduino (https://www.pjrc.com/store/teensy40.html)

## Usage
```bash
python train_sac.py 
```

## Description of Codes
- `environment.py` -> real-world RL environment 
- `replayBuffer.py` -> replay buffer implementation
- `train_sac.py` -> entry point for training 
- `models`
  - `sac_model.py`  -> neural network implementations used for SAC algorithm
- `utils`
  - `dwclient.py`  -> client code for communication between the drone and the algorithm server
  - `dwserver.py`  -> server code for communication between the drone and the algorithm server
  - `NatNetClient.py` -> Code for Opti track motion capture data streaming
  - `serialChannel.py` -> Code for Teensyduino serial communication
  - `util` -> code for saving trained model and logs


## Dependencies
1. torch	2.0.1	
2. matplotlib	3.7.1
3. numpy	1.23.2
4. pandas	2.0.3
   

