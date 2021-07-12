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

def write_summary(writer, n_epi, score, optimization_step, avg_loss, epsilon, env:Environment, avg_loop_t, train_t):
    writer.add_scalar('agent/score', score, n_epi)
    writer.add_scalar('agent/eps', epsilon, n_epi)
    writer.add_scalar('agent/avg_angle', env.angle_sum/float(env.step_count), n_epi)
    writer.add_scalar('agent/len_epi', env.step_count, n_epi)
    writer.add_scalar('train/step', optimization_step, n_epi)
    writer.add_scalar('train/loss', avg_loss, n_epi)
    writer.add_scalar('time/loop', avg_loop_t, n_epi)
    writer.add_scalar('time/train', train_t, n_epi)
    writer.add_scalar('state/max_val', env.max_s, n_epi)
    writer.add_scalar('state/min_val', env.min_s, n_epi)
    writer.add_scalar('action/ratio_motor_power_0', env.action_count[0] / float(np.sum(env.action_count)), n_epi)
    writer.add_scalar('action/ratio_motor_power_150', env.action_count[1] / float(np.sum(env.action_count)), n_epi)
    writer.add_scalar('action/ratio_motor_power_200', env.action_count[2] / float(np.sum(env.action_count)), n_epi)
    writer.add_scalar('action/ratio_motor_power_250', env.action_count[3] / float(np.sum(env.action_count)), n_epi)
