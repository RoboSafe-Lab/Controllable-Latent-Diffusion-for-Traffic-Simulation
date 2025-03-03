
import torch.optim as optim
import torch
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
import math
from models.rl.criticmodel import compute_reward,ReplayBuffer,failure_rate_compute
from torch.optim.lr_scheduler import LambdaLR
import tbsim.utils.tensor_utils as TensorUtils
from tqdm.auto import tqdm
from memory_profiler import profile
from rich.progress import Progress
from scipy.stats import wasserstein_distance
import numpy as np

class GuideDMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config,modality_shapes,ckpt_dm):

        super(GuideDMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
        self.epochs = train_config.training.epochs
        
       
        self.num_samp = algo_config.num_samp
        self.ppo_mini_batch = algo_config.ppo_mini_batch
        self.dm = DmModel(algo_config,modality_shapes)
        if ckpt_dm is not None:
            ckpt = torch.load(ckpt_dm,map_location='cpu')
            dm_state = {}
            prefix = 'dm.'

            for old_key, value in ckpt['state_dict'].items():
                if old_key.startswith(prefix):
                    new_key = old_key[len(prefix):]
                    dm_state[new_key]=value
            missing, unexpected = self.dm.load_state_dict(dm_state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.vae = VaeModel(algo_config,train_config, modality_shapes)
        for param in self.vae.lstmvae.parameters():
            param.requires_grad = False

        self.buffer_max = algo_config.buffer_max
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_max)
        self.ppo_update_times = algo_config.ppo_update_times

        self.update_interval = algo_config.update_interval
        self.steps_since_update=0

        self.automatic_optimization = False


        self.dt = algo_config.step_time
    def configure_optimizers(self):  

        optim_params_dm = self.algo_config.optim_params["dm"]
        optimizer = optim.Adam(
            params=self.dm.parameters(),
            lr=optim_params_dm["learning_rate"]["initial"],
            weight_decay=optim_params_dm["regularization"]["L2"],
        )
        
        total_epochs = self.epochs
        warmup_epoch = total_epochs / 3

        def lr_lambda(epoch):
            if epoch<warmup_epoch:
                return float(epoch)/float(max(1,warmup_epoch))
            else:
                progress = float(epoch-warmup_epoch)/float(max(1,total_epochs-warmup_epoch))
                return 0.5*(1. + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer,lr_lambda)
        return [optimizer],[{
            'scheduler':scheduler,
            'interval':"epoch",
            'frequency':1,
            'name':'warmup_cosine_lr'
        }]
      
    def training_step(self, batch):
        batch = batch_utils().parse_batch(batch) 
        aux_info, _,_ = self.vae.pre_vae(batch)
        out_dict = self.dm(batch,aux_info,self.algo_config)

        x1 = out_dict['x1']         
        x0 = out_dict['pred_traj']  
        log_prob_old = out_dict['log_prob_final']

        aux_info = out_dict['aux_info']
        cond_feat_value = aux_info['cond_feat'].detach().cpu() if isinstance(aux_info['cond_feat'], torch.Tensor) else aux_info['cond_feat']


        action_decoder = self.vae.lstmvae.lstm_dec(x0,aux_info['cond_feat'])
        recon_state_and_action_descaled = self.vae.convert_action_to_state_and_action(action_decoder,aux_info['curr_states'],descaled_output=True)
      
        state_descaled = recon_state_and_action_descaled[...,:2] 
        state_descaled = TensorUtils.reshape_dimensions(state_descaled,begin_axis=0,end_axis=1,target_dims=(self.batch_size,self.num_samp))
        recon_state_and_action_descaled = TensorUtils.reshape_dimensions(recon_state_and_action_descaled,begin_axis=0,end_axis=1,target_dims=(self.batch_size,self.num_samp))
        recon_state_and_action_scaled = self.vae.scale_traj(recon_state_and_action_descaled)
        reward = compute_reward(recon_state_and_action_descaled,batch,recon_state_and_action_scaled) 

        self.replay_buffer.add(x0,x1,log_prob_old,reward,cond_feat_value)
        
       
        vis_dict={
            "raster_from_agent":batch['raster_from_agent'],
            "image":batch['image'],
            'traj':state_descaled,
        }
        self.steps_since_update += 1
        if self.steps_since_update < self.update_interval:
            self.log('train/reward', reward.mean(),batch_size=self.batch_size,prog_bar=True)
            
            return vis_dict
        
        else:
            ppo_loss = self.ppo_update()
            self.log('train/ppo_loss', ppo_loss.item(), batch_size=self.batch_size,prog_bar=True)
            self.log('train/reward', reward.mean(),batch_size=self.batch_size,prog_bar=True)
            self.steps_since_update = 0
            return vis_dict
   
    def ppo_update(self):
        ppo_epochs = 10  
        eps = 0.2
        losses = []
        opt = self.optimizers()
        
        with Progress(transient=True) as progress:
            epoch_task = progress.add_task("[green]Epoch Progress...", total=ppo_epochs)
            for epoch in range(ppo_epochs):
                
                batch_task = progress.add_task("[blue]Mini-Batch Progress...", total=self.ppo_update_times)
                for _ in range(self.ppo_update_times):
                   
                    batch = self.replay_buffer.sample(self.ppo_mini_batch)
                    
                    x0_list, x1_list, log_p_old_list, reward_list, cond_feat_value = zip(*batch)
                    
                  
                    x0_batch = torch.stack(x0_list).to(self.device)
                    x0_batch = x0_batch.view(-1, x0_batch.size(-2), x0_batch.size(-1)) 

                    x1_batch = torch.stack(x1_list).to(self.device)
                    x1_batch = x1_batch.view(-1, x1_batch.size(-2), x1_batch.size(-1))  

                    log_p_old_batch = torch.stack(log_p_old_list).to(self.device).view(-1)  
                    reward_batch = torch.stack(reward_list).to(self.device).view(-1)  
                    
            
                    baseline = self.replay_buffer.get_baseline()
                    advantage = reward_batch - baseline 
                    
    
                    aux_info_stacked = torch.stack(cond_feat_value, dim=0).to(self.device)
                   
                  
                    t_tensor = torch.zeros(x0_batch.size(0), device=self.device, dtype=torch.long)
                   
                    log_p_new = self.dm.log_prob(x1_batch, x0_batch, {'cond_feat': aux_info_stacked}, t=t_tensor)
                    ratios = torch.exp(log_p_new - log_p_old_batch)
                    
                    surr1 = ratios * advantage
                    surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantage
                    loss = -torch.min(surr1, surr2).mean()
                    losses.append(loss.detach())
                    
                    opt.zero_grad()
                    self.manual_backward(loss)
                    opt.step()
                    
                    progress.update(batch_task, advance=1)
                    del batch, x0_list, x1_list, log_p_old_list, reward_list, cond_feat_value
                    del x0_batch, x1_batch, log_p_old_batch, reward_batch, advantage, aux_info_stacked, t_tensor, log_p_new, ratios, surr1, surr2, loss
                progress.update(epoch_task, advance=1)
        
        mean_loss = torch.stack(losses).mean()
        losses.clear() 
        return mean_loss

      
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch) 
        aux_info, _,_ = self.vae.pre_vae(batch)
        out_dict = self.dm(batch,aux_info,self.algo_config)

        x0 = out_dict['pred_traj']  
        
        aux_info = out_dict['aux_info']

        action_decoder = self.vae.lstmvae.lstm_dec(x0,aux_info['cond_feat'])
        recon_state_and_action_descaled = self.vae.convert_action_to_state_and_action(action_decoder,aux_info['curr_states'],descaled_output=True)
      
        state_descaled = recon_state_and_action_descaled[...,:2] 
        state_descaled = TensorUtils.reshape_dimensions(state_descaled,begin_axis=0,end_axis=1,target_dims=(self.batch_size,self.num_samp))

        reward = compute_reward(state_descaled,batch).mean() 
        self.log('val/reward',reward,batch_size=self.batch_size,prog_bar=True)
        
    def test_step(self, batch):
        batch = batch_utils().parse_batch(batch) 
        aux_info, state_and_action,_ = self.vae.pre_vae(batch)
        out_dict = self.dm(batch,aux_info,self.algo_config)
        x0 = out_dict['pred_traj']  
        aux_info = out_dict['aux_info']
        action_decoder = self.vae.lstmvae.lstm_dec(x0,aux_info['cond_feat'])
        recon_state_and_action_descaled = self.vae.convert_action_to_state_and_action(action_decoder,aux_info['curr_states'],descaled_output=True) 

        stats = failure_rate_compute(recon_state_and_action_descaled,batch)
        self.all_failure_stats = []
        self.all_failure_stats.append(stats)


        recon_state_and_action_scaled = self.vae.scale_traj(recon_state_and_action_descaled)
        dt = self.dt

        
        long_acc_gt = state_and_action[...,4] 
        long_acc_pred = recon_state_and_action_scaled[...,4]

        
        lat_acc_gt = state_and_action[..., 2] * state_and_action[..., 5]
        lat_acc_pred = recon_state_and_action_scaled[..., 2] * recon_state_and_action_scaled[..., 5]

     
        jerk_gt = (long_acc_gt[:, 1:] - long_acc_gt[:, :-1]) / dt
        jerk_pred = (long_acc_pred[:, 1:] - long_acc_pred[:, :-1]) / dt

        long_acc_gt_flat = long_acc_gt.detach().cpu().numpy().flatten()
        long_acc_pred_flat = long_acc_pred.detach().cpu().numpy().flatten()
        lat_acc_gt_flat = lat_acc_gt.detach().cpu().numpy().flatten()
        lat_acc_pred_flat = lat_acc_pred.detach().cpu().numpy().flatten()
        jerk_gt_flat = jerk_gt.detach().cpu().numpy().flatten()
        jerk_pred_flat = jerk_pred.detach().cpu().numpy().flatten()

        self.test_outputs = []
        self.test_outputs.append({
                'long_acc_gt': long_acc_gt_flat,
                'long_acc_pred': long_acc_pred_flat,
                'lat_acc_gt': lat_acc_gt_flat,
                'lat_acc_pred': lat_acc_pred_flat,
                'jerk_gt': jerk_gt_flat,
                'jerk_pred': jerk_pred_flat,
            })
      

        return {}

    def on_test_epoch_end(self):
        offroad_rates = [stat['offroad_failure_rate'] for stat in self.all_failure_stats]
        collision_rates = [stat['collision_failure_rate'] for stat in self.all_failure_stats]
        overall_rates = [stat['overall_failure_rate'] for stat in self.all_failure_stats]
        
        avg_offroad_failure_rate = sum(offroad_rates) / len(offroad_rates)
        avg_collision_failure_rate = sum(collision_rates) / len(collision_rates)
        avg_overall_failure_rate = sum(overall_rates) / len(overall_rates)
        
        self.log("avg_offroad_failure_rate", avg_offroad_failure_rate, prog_bar=True)
        self.log("avg_collision_failure_rate", avg_collision_failure_rate, prog_bar=True)
        self.log("avg_overall_failure_rate", avg_overall_failure_rate, prog_bar=True)
        
        print(f"Epoch Average Offroad Failure Rate: {avg_offroad_failure_rate:.4f}")
        print(f"Epoch Average Collision Failure Rate: {avg_collision_failure_rate:.4f}")
        print(f"Epoch Average Overall Failure Rate: {avg_overall_failure_rate:.4f}")
        outputs = self.test_outputs
        long_acc_gt_all = np.concatenate([x['long_acc_gt'] for x in outputs])
        long_acc_pred_all = np.concatenate([x['long_acc_pred'] for x in outputs])
        lat_acc_gt_all = np.concatenate([x['lat_acc_gt'] for x in outputs])
        lat_acc_pred_all = np.concatenate([x['lat_acc_pred'] for x in outputs])
        jerk_gt_all = np.concatenate([x['jerk_gt'] for x in outputs])
        jerk_pred_all = np.concatenate([x['jerk_pred'] for x in outputs])

        wd_long = wasserstein_distance(long_acc_gt_all, long_acc_pred_all)
        wd_lat = wasserstein_distance(lat_acc_gt_all, lat_acc_pred_all)
        wd_jerk = wasserstein_distance(jerk_gt_all, jerk_pred_all)
        
        realism_deviation = (wd_long + wd_lat + wd_jerk) / 3.0

        self.log('wd_long', wd_long, prog_bar=True)
        self.log('wd_lat', wd_lat, prog_bar=True)
        self.log('wd_jerk', wd_jerk, prog_bar=True)
        self.log('realism_deviation', realism_deviation, prog_bar=True)

        print(f"Test Epoch: wd_long={wd_long:.4f}, wd_lat={wd_lat:.4f}, wd_jerk={wd_jerk:.4f}, realism_deviation={realism_deviation:.4f}")

        return {
        'wd_long': wd_long,
        'wd_lat': wd_lat,
        'wd_jerk': wd_jerk,
        'realism_deviation': realism_deviation,
    }
        
   
    def on_save_checkpoint(self, checkpoint):
        full_sd = super().state_dict()

        dm_only_sd = {}
        for k, v in full_sd.items():
            if k.startswith("dm."):
                dm_only_sd[k] = v
    
        checkpoint["state_dict"] = dm_only_sd
 