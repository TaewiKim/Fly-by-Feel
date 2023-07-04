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

def save_sac_model(config, q1, q2, pi, n_epi):
    model_dict = {
        'optimization_step': pi.optimization_step,
        'n_epi': n_epi,
        'q1_state_dict': q1.state_dict(),
        'q2_state_dict': q2.state_dict(),
        'pi_state_dict': pi.state_dict(),

        'q1_optimizer_state_dict': q1.optimizer.state_dict(),
        'q2_optimizer_state_dict': q2.optimizer.state_dict(),
        'pi_optimizer_state_dict': pi.optimizer.state_dict(),
        'alpha_optimizer_state_dict': pi.log_alpha_optimizer.state_dict(),
    }
    path = config["log_dir"] + f"/sac_model_{pi.optimization_step}.tar"
    torch.save(model_dict, path)

def save_dqn_model(config, model, n_epi):
    model_dict = {
        'optimization_step': model.optimization_step,
        'n_epi' : n_epi,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }
    path = config["log_dir"] + f"/dqn_model_{model.optimization_step}.tar"
    torch.save(model_dict, path)

def write_summary(writer, config, n_epi, score, optimization_step, avg_loss, epsilon, env:Environment, avg_loop_t, train_t, alpha, action_sum, entropy):
    writer.add_scalar('agent/score', score, n_epi)
    writer.add_scalar('agent/avg_Px', env.Px_sum/float(env.step_count), n_epi)
    writer.add_scalar('agent/max_Px', env.max_Px, n_epi)
    writer.add_scalar('agent/min_Px', env.min_Px, n_epi)
    writer.add_scalar('agent/avg_Py', env.Py_sum/float(env.step_count), n_epi)
    writer.add_scalar('agent/max_Py', env.max_Py, n_epi)
    writer.add_scalar('agent/min_Py', env.min_Py, n_epi)
    writer.add_scalar('agent/avg_Pz', env.Pz_sum/float(env.step_count), n_epi)
    writer.add_scalar('agent/max_Pz', env.max_Pz, n_epi)
    writer.add_scalar('agent/min_Pz', env.min_Pz, n_epi)

    writer.add_scalar('agent/avg_actions_thrust', action_sum[0] / float(env.step_count), n_epi)
    writer.add_scalar('agent/avg_actions_direction', action_sum[1] / float(env.step_count), n_epi)

    writer.add_scalar('agent/len_epi', env.step_count, n_epi)
    writer.add_scalar('train/alpha', alpha, n_epi)
    writer.add_scalar('train/step', optimization_step, n_epi)
    writer.add_scalar('train/loss', avg_loss, n_epi)
    writer.add_scalar('train/entropy', entropy, n_epi)
    writer.add_scalar('time/loop', avg_loop_t, n_epi)
    writer.add_scalar('time/train', train_t, n_epi)
    writer.add_scalar('state/max_val', env.max_s, n_epi)
    writer.add_scalar('state/min_val', env.min_s, n_epi)
