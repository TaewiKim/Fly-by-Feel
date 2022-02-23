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

def write_summary(writer, config, n_epi, score, optimization_step, avg_loss, epsilon, env:Environment, avg_loop_t, train_t, alpha, action_sum):
    writer.add_scalar('agent/score', score, n_epi)
    writer.add_scalar('agent/avg_angle', env.angle_sum/float(env.step_count), n_epi)
    writer.add_scalar('agent/max_angle', env.max_angle, n_epi)
    writer.add_scalar('agent/min_angle', env.min_angle, n_epi)
    writer.add_scalar('agent/avg_actions', action_sum / float(env.step_count), n_epi)

    # writer.add_scalar('distance/min', np.min(env.distance), n_epi)
    # writer.add_scalar('distance/mean', np.mean(env.distance), n_epi)
    # writer.add_scalar('distance/max', np.max(env.distance), n_epi)
    # writer.add_scalar('distance/x_min', np.min(env.x_lst), n_epi)
    # writer.add_scalar('distance/x_mean', np.mean(env.x_lst), n_epi)
    # writer.add_scalar('distance/x_max', np.max(env.x_lst), n_epi)
    # writer.add_scalar('distance/y_min', np.min(env.y_lst), n_epi)
    # writer.add_scalar('distance/y_mean', np.mean(env.y_lst), n_epi)
    # writer.add_scalar('distance/y_max', np.max(env.y_lst), n_epi)
    # writer.add_scalar('distance/z_min', np.min(env.z_lst), n_epi)
    # writer.add_scalar('distance/z_mean', np.mean(env.z_lst), n_epi)
    # writer.add_scalar('distance/z_max', np.max(env.z_lst), n_epi)

    writer.add_scalar('agent/len_epi', env.step_count, n_epi)
    writer.add_scalar('train/alpha', alpha, n_epi)
    writer.add_scalar('train/step', optimization_step, n_epi)
    writer.add_scalar('train/loss', avg_loss, n_epi)
    writer.add_scalar('time/loop', avg_loop_t, n_epi)
    writer.add_scalar('time/train', train_t, n_epi)
    writer.add_scalar('state/max_val', env.max_s, n_epi)
    writer.add_scalar('state/min_val', env.min_s, n_epi)
    # if config["is_discrete"]:
    #     writer.add_scalar('agent/eps', epsilon, n_epi)
    #     writer.add_scalar('action/ratio_motor_power_0', env.action_count[0] / float(np.sum(env.action_count)), n_epi)
    #     writer.add_scalar('action/ratio_motor_power_150', env.action_count[1] / float(np.sum(env.action_count)), n_epi)
    #     writer.add_scalar('action/ratio_motor_power_200', env.action_count[2] / float(np.sum(env.action_count)), n_epi)
    #     writer.add_scalar('action/ratio_motor_power_250', env.action_count[3] / float(np.sum(env.action_count)), n_epi)
