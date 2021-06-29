import os, json
import torch
from environment import Environment
import numpy as np

def save_config(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"]+"/args.json","w")
    f.write(args_info)
    f.close()

def save_model(config, model):
    model_dict = {
        'optimization_step': model.optimization_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }

    path = config["log_dir"] + f"/model_{model.optimization_step}.tar"
    torch.save(model_dict, path)

def write_summary(writer, n_epi, score, optimization_step, avg_loss, epsilon, env:Environment):
    writer.add_scalar('agent/score', score, n_epi)
    writer.add_scalar('agent/eps', epsilon, n_epi)
    writer.add_scalar('agent/avg_angle', env.angle_sum/float(env.step_count), n_epi)
    writer.add_scalar('agent/len_epi', env.step_count, n_epi)
    writer.add_scalar('agent/power_0', env.action_count[0]/float(np.sum(env.action_count)), n_epi)
    writer.add_scalar('agent/power_max', env.action_count[2] / float(np.sum(env.action_count)), n_epi)
    writer.add_scalar('train/step', optimization_step, n_epi)
    writer.add_scalar('train/loss', avg_loss, n_epi)