from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment import Environment, DummyEnv
from models.sac_model import PolicyNet, QNet, calc_target
import torch
from replayBuffer import ReplayBuffer
import time, os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils.util import save_config, save_sac_model as save_model, write_summary
import numpy as np
import pandas as pd
import random
from utils.NatNetClient import NatNetClient

def main():
    s_channel = serialPlot('COM27', 19200, 4)  # dataNumBytes 4  : number of bytes of 1 data point
    s_channel.readSerialStart()  # starts background thread
    config = {"target_position": []}
    my_thread = []
    streamingClient = []
    env = Environment(config, my_thread, s_channel, streamingClient)
    env.drone_shoot()
    time.sleep(1)
    env.shooter_back()

main()
