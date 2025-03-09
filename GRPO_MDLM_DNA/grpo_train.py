# GRPO
import diffusion_gosai_update_new
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
import torch
import argparse
import wandb
import os
import datetime
from utils import str2bool, set_seed

def fine_tune(new_model, reward_model, old_model, ref_model, args):

    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')
            
    eps = 1e-5
    eps_error = 1e-8
    dt = (1 - eps) / args.total_num_steps
    current_kl_coeff = args.kl_coeff
    current_entropy_coeff = args.entropy_coeff
    
    new_model.train()
    torch.set_grad_enabled(True)
    optimizer = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    
    batch_losses = []
    batch_rewards = []
    for epoch_num in range(args.num_epochs):
        
        print(f"------ Ref model copy param ------")
        # Copy parameters from the new model to the ref model
        ref_model.load_state_dict(new_model.state_dict())
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False 
        
        print(f"-----Epoch {epoch_num} ------")
        
        for _step in range(args.num_steps_grpo):
            current_full_step = args.num_steps_grpo*(epoch_num)+_step
            print(f"-----Step {current_full_step}, epoch {epoch_num} ------")
            rewards = []
            rewards_eval = []
            losses = []
            reward_losses = []
            kl_losses = []
            entropy_losses = []
            tot_grad_norm = 0.0

            ### Copy parameters from the new model to the old model
            print(f"------ Old model copy param ------")
            old_model.load_state_dict(new_model.state_dict())
            old_model.eval()
            for param in old_model.parameters():
                param.requires_grad = False 

            ### Sampling group of outputs
            x_olds = []
            xr_olds = []
            for _ in range(args.grpo):
                x_old, _, _, _, _ = old_model._sample_no_gradient(eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp) # [bsz, seqlen, 4]
                x_olds.append(x_old)
                xr_old = torch.argmax(x_old, dim=-1)
                xr_olds.append(xr_old)

            ### compute group of rewards
            reward_list = []
            for x_old in x_olds:
                x_argmax_old = torch.transpose(x_old, 1, 2)
                preds = reward_model(x_argmax_old).squeeze(-1) # [bsz, 3]
                reward = preds[:, 0].detach()
                reward_list.append(reward)
            
            ### computing the advantages
            stacked_tenseurs = torch.stack(reward_list)
            mean_ = stacked_tenseurs.mean(dim=0, keepdim=True)
            std_ = stacked_tenseurs.std(dim=0, keepdim=True)
            normalised_tenseurs = (stacked_tenseurs - mean_) / (std_ + eps_error)
            reward_list = list(normalised_tenseurs)
            
            ### GRPO losses
            for mu_ in range(args.mu_grpo):
                
                ### reward eval
                x, last_x_list, condt_list, move_chance_t_list, copy_flag_list = new_model._sample_finetune_gradient(eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp)
                x_argmax = torch.transpose(x, 1, 2)
                preds_argmax = reward_model(x_argmax).squeeze(-1)
                reward_argmax = preds_argmax[:, 0]
                rewards_eval.append(reward_argmax.detach().cpu().numpy())
                
                xr = torch.argmax(x, dim=-1)
                
                reward_loss = 0.0
                entropy_loss = 0.0
                for i, xr_old in enumerate(xr_olds):
                    s_t_old_x_z = old_model.forward_at_time(xr_old, dt, full_scores=True).exp().detach()
                    pi_old_z = old_model._build_distrib(xr_old, dt* torch.ones(x_old.shape[0], 1, device=old_model.device), dt, full_scores=True).detach()
                    pi_old_z[:, :, 4] = eps_error
                    
                    reward = reward_list[i]
                    rewards.append(reward.detach().cpu().numpy())

                    reward_loss = reward_loss + new_model._compute_loss_sepo(reward, s_t_old_x_z, pi_old_z, xr_old, t=dt, epsilon=args.eps_grpo, full_scores=True)
                        
                reward_loss = - reward_loss / args.grpo # The minus sign is correct here, check the paper if not conviced !
                
                ### potential KL term
                if args.put_kl:
    
                    total_kl = []
                    
                    # calculate the KL divergence
                    for random_t in range(args.total_num_steps):
                        if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
                            continue
                        last_x = last_x_list[random_t] # [bsz, seqlen, 5]
                        condt = condt_list[random_t]
                        move_chance_t = move_chance_t_list[random_t]
                        copy_flag = copy_flag_list[random_t] # [bsz, seqlen, 1]
                        log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]
                        log_p_x0_old = ref_model.forward(last_x, condt)[:, :, :-1]
        
                        p_x0 = log_p_x0.exp() # [bsz, seqlen, 4]
                        p_x0_old = log_p_x0_old.exp()
        
                        kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0,0,0]
                        kl_div = ((kl_div * last_x[:, :, :-1]).sum((1, 2))).clone() # [bsz]
                        total_kl.append(kl_div)
                        
                    if current_full_step < args.kl_coeff_schedule_warmup:
                        # linear warmup
                        current_kl_coeff = (epoch_num + 1) / args.kl_coeff_schedule_warmup * args.kl_coeff
                    else:
                        current_kl_coeff = args.kl_coeff
                    
                    kl_loss = torch.stack(total_kl, 1).sum(1).mean()
                else :
                    current_kl_coeff = args.kl_coeff
                    kl_loss = torch.tensor([0.0], device=new_model.device)
                    
                ### potential entropy bonus term, to encourage exploration
                if args.entropy:
                    entropy_loss += new_model._compute_entropy_bonus(xr, t=dt, full_scores=True, eps_error=eps_error).clone()
                    if  current_full_step < args.entropy_coeff_schedule_warmup:
                        # linear warmup
                        current_entropy_coeff = (epoch_num + 1) / args.kl_coeff_schedule_warmup * args.entropy_coeff
                    else:
                        current_entropy_coeff = args.entropy_coeff
                else:
                    entropy_loss = torch.tensor([0.0], device=new_model.device)
                
                loss = -reward_loss + current_kl_coeff*kl_loss - current_entropy_coeff*entropy_loss #we minimize this loss, so we minimize the **negative** reward, we minimize the **positive** kl and we minimize the **negative** entropy bonus.
                loss.backward(retain_graph=True)
                
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
                tot_grad_norm += norm
                optimizer.step()
                optimizer.zero_grad()
    
                batch_losses.append(loss.cpu().detach().numpy())
                batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
                losses.append(loss.cpu().detach().numpy())
                reward_losses.append(reward_loss.cpu().detach().numpy())
                kl_losses.append(current_kl_coeff*kl_loss.cpu().detach().numpy())
                entropy_losses.append(current_entropy_coeff*entropy_loss.cpu().detach().numpy())
            
            
            rewards = np.array(rewards)
            rewards_eval = np.array(rewards_eval)
            losses = np.array(losses)
            reward_losses = np.array(reward_losses)
            kl_losses = np.array(kl_losses)
            entropy_losses = np.array(entropy_losses)

            print("Epoch %d"%epoch_num, "Mean reward %f"%np.mean(rewards), "Mean reward eval %f"%np.mean(rewards_eval), 
            "Mean grad norm %f"%tot_grad_norm, "Mean loss %f"%np.mean(losses), "Mean reward loss %f"%np.mean(reward_losses), "Mean kl loss %f"%np.mean(kl_losses), "Mean entropy loss %f"%np.mean(entropy_losses))
            if args.name != 'debug':
                wandb.log({"epoch": epoch_num, "mean_reward": np.mean(rewards), "mean_reward_eval": np.mean(rewards_eval), 
                "mean_grad_norm": tot_grad_norm, "mean_loss": np.mean(losses), "mean reward loss": np.mean(reward_losses), "mean kl loss": np.mean(kl_losses), "mean entropy loss": np.mean(entropy_losses)})
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch_num} Mean reward {np.mean(rewards)} Mean reward eval {np.mean(rewards_eval)} Mean grad norm {tot_grad_norm} Mean loss {np.mean(losses)} Mean reward loss {np.mean(reward_losses)} Mean kl loss {np.mean(kl_losses)} Mean entropy loss {np.mean(entropy_losses)}\n")
            
            if (_step+1) % args.save_every_n_steps == 0:
                model_path = os.path.join(save_path, f'model_step{args.num_steps_grpo*(epoch_num)+_step+1}.ckpt')
                torch.save(new_model.state_dict(), model_path)
                print(f"Model saved at epoch {epoch_num}")

    if args.name != 'debug':
        wandb.finish()

    return batch_losses

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General run arguments
argparser.add_argument('--base_path', type=str, default='data/scratch/wangchy/seqft/',
                        help="Base directory where model checkpoints and logs are stored.")
argparser.add_argument('--name', type=str, default='date_and_time',
                        help="Experiment name. Default is the current date and time.")
argparser.add_argument('--save_every_n_steps', type=int, default=10,
                        help="Number of training steps between saving model checkpoints.")
argparser.add_argument("--gpu_number", type=int, default=0,
                        help="GPU device ID to use for training.")
argparser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
argparser.add_argument('--batch_size', type=int, default=16,
                        help="Number of samples per batch during training.")
argparser.add_argument('--total_num_steps', type=int, default=128,
                        help="Total number of training steps.")
argparser.add_argument('--copy_flag_temp', type=float, default=None,
                        help="Temperature parameter for copy flag mechanism. (If applicable)")

# Optimizer arguments
argparser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Initial learning rate for the optimizer.")
argparser.add_argument('--wd', type=float, default=0.0,
                        help="Weight decay (L2 regularization) applied to optimizer.")
argparser.add_argument('--gradnorm_clip', type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping.")

# KL-divergence arguments
argparser.add_argument("--put_kl", type=int, default=0,
                        help="Enable KL-divergence regularization (1 for True, 0 for False).")
argparser.add_argument("--truncate_kl", type=str2bool, default=True,
                        help="Whether to truncate KL-divergence calculations.")
argparser.add_argument('--truncate_steps', type=int, default=50,
                        help="Number of steps after which KL divergence is truncated.")
argparser.add_argument('--kl_coeff', type=float, default=1e-4,
                        help="Weighting coefficient for KL-divergence loss.")
argparser.add_argument('--kl_coeff_schedule_warmup', type=int, default=0,
                        help="Number of warm-up steps for KL coefficient scheduling.")

# Entropy bonus arguments
argparser.add_argument('--entropy', type=str2bool, default=False,
                        help="Enable entropy bonus in the loss function.")
argparser.add_argument('--entropy_coeff', type=float, default=1e-5,
                        help="Coefficient for entropy regularization.")
argparser.add_argument('--entropy_coeff_schedule_warmup', type=int, default=0,
                        help="Number of warm-up steps for entropy coefficient scheduling.")

#GRPO args
argparser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of training epochs.")
argparser.add_argument('--num_steps_grpo', type=int, default=10,
                        help="Number of steps per GRPO update.")
argparser.add_argument("--grpo", type=int, default=8,
                        help="GRPO hyperparameter controlling policy update size.")
argparser.add_argument("--mu_grpo", type=int, default=5,
                        help="GRPO smoothing parameter.")
argparser.add_argument('--eps_grpo', type=float, default=0.2,
                        help="Epsilon parameter for GRPO.")

args = argparser.parse_args()
print(args)


# pretrained model path
CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
log_base_dir = os.path.join(args.base_path, 'mdlm/reward_bp_results_final')

# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs_gosai", job_name="load_model")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
prefix = f"R"
if args.put_kl:
    prefix+=f"K{args.kl_coeff}"
if args.entropy:
    prefix+=f"H{args.entropy_coeff}"

# initialize a log file
if args.name == 'debug':
    print("Debug mode")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    run_name = f'{prefix}_bsz{args.batch_size}_lr{args.learning_rate}_grpo{args.grpo}_steps_grpo{args.num_steps_grpo}_mu{args.mu_grpo}_clip{args.gradnorm_clip}_{args.name}{curr_time}'
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project='reward_bp_final', name=run_name, config=args, dir=save_path)
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

new_model = diffusion_gosai_update_new.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg).to(f"cuda:{args.gpu_number}")
old_model = diffusion_gosai_update_new.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg).to(f"cuda:{args.gpu_number}")
ref_model = diffusion_gosai_update_new.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg).to(f"cuda:{args.gpu_number}")

reward_model = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
reward_model.eval()

fine_tune(new_model, reward_model, old_model, ref_model, args)