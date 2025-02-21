
import torch.optim as optim
import torch
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
import math
from models.rl.criticmodel import compute_reward,ReplayBuffer
from torch.optim.lr_scheduler import LambdaLR
import tbsim.utils.tensor_utils as TensorUtils
from tqdm.auto import tqdm
from alive_progress import alive_bar
from rich.progress import Progress


class GuideDMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config,modality_shapes,ckpt_dm):

        super(GuideDMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
        self.epochs = train_config.training.epochs
        
       
        self.num_samp = algo_config.num_samp
        self.ppo_sample = algo_config.ppo_num
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
    def configure_optimizers(self):  

        optim_params_dm = self.algo_config.optim_params["dm"]
        optimizer = optim.Adam(
            params=self.dm.parameters(),
            lr=optim_params_dm["learning_rate"]["initial"],
            weight_decay=optim_params_dm["regularization"]["L2"],
        )
        warmup_epoch=10
        total_epochs = self.epochs

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

        x1 = out_dict['x1']         # shape: [B* N, T, latent_size]
        x0 = out_dict['pred_traj']  # shape: [B* N, T, latent_size]
        log_prob_old = out_dict['log_prob_final']#[B*N]

        aux_info = out_dict['aux_info']
        aux_info_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in aux_info.items()}

        action_decoder = self.vae.lstmvae.lstm_dec(x0,aux_info['cond_feat'])#[B*N,52,2]
        recon_state_and_action_descaled = self.vae.convert_action_to_state_and_action(action_decoder,aux_info['curr_states'],descaled_output=True)
      
        state_descaled = recon_state_and_action_descaled[...,:2] #[B*N,52,2]
        state_descaled = TensorUtils.reshape_dimensions(state_descaled,begin_axis=0,end_axis=1,target_dims=(self.batch_size,self.num_samp))#[B,N,52,2]

        reward = compute_reward(state_descaled,batch) #[B, N]

        # aux_info_detached = detach_aux_info(aux_info)
        self.replay_buffer.add(x0,x1,log_prob_old,reward,aux_info_cpu)
        self.steps_since_update += 1
       
        vis_dict={
            "raster_from_agent":batch['raster_from_agent'],
            "image":batch['image'],
            'traj':state_descaled,
        }
        
        # if len(self.replay_buffer) < self.buffer_max or self.steps_since_update < self.update_interval:
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
        
        eps = 0.2
        losses = []
        opt = self.optimizers()
        # for _ in tqdm(range(self.ppo_update_times),desc='PPO Update',leave=False):
        with Progress() as progress:
            task = progress.add_task("[green]PPO Update...",total=self.ppo_update_times)
            for _ in range(self.ppo_update_times):
                batch = self.replay_buffer.sample(self.ppo_sample)
                x0_batch, x1_batch, log_p_old_batch, reward, aux_info_batch= zip(*batch)
                
                x0_batch = torch.stack(x0_batch).to(self.device)
                x0_batch = x0_batch.view(-1, x0_batch.size(-2), x0_batch.size(-1)) #[M, 52, 4]

                x1_batch = torch.stack(x1_batch).to(self.device)
                x1_batch = x1_batch.view(-1, x1_batch.size(-2), x1_batch.size(-1)) #[M,52,4]

                log_p_old_batch = torch.stack(log_p_old_batch).to(self.device)
                log_p_old_batch = log_p_old_batch.view(-1) #[M]

                reward_batch = torch.stack(reward).to(self.device)
                reward_batch = reward_batch.view(-1) #[M]
                

                baseline = self.replay_buffer.get_baseline()
                advantage = reward_batch - baseline #[M]

                # cond_feat_list = [d['cond_feat'] for d in aux_info_batch]
                # aux_info_stacked = torch.stack(cond_feat_list, dim=0).view(-1, cond_feat_list[0].shape[-1]).to(self.device)

                aux_info_stacked = torch.stack([d['cond_feat'] for d in aux_info_batch], dim=0).view(-1, [d['cond_feat'] for d in aux_info_batch][0].shape[-1]).to(self.device) #[M,256]

                log_p_new = self.dm.log_prob(x1_batch, x0_batch, {'cond_feat':aux_info_stacked}, t=torch.zeros(x0_batch.shape[0], device=self.device, dtype=torch.long))#[M*B*N]
                ratios = torch.exp(log_p_new - log_p_old_batch)


                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantage
                loss = -torch.min(surr1, surr2)
                loss = loss.mean()
                losses.append(loss)

            
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
                progress.update(task, advance=1)
        losses_tensor = torch.stack(losses)
        mean_loss = losses_tensor.mean()
        return mean_loss
      
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch) 
        aux_info, _,_ = self.vae.pre_vae(batch)
        out_dict = self.dm(batch,aux_info,self.algo_config)

        x0 = out_dict['pred_traj']  # shape: [B* num_samp, horizon, latent_size]
        
        aux_info = out_dict['aux_info']

        action_decoder = self.vae.lstmvae.lstm_dec(x0,aux_info['cond_feat'])#[B*N,52,2]
        recon_state_and_action_descaled = self.vae.convert_action_to_state_and_action(action_decoder,aux_info['curr_states'],descaled_output=True)
      
        state_descaled = recon_state_and_action_descaled[...,:2] #[B*N,52,2]
        state_descaled = TensorUtils.reshape_dimensions(state_descaled,begin_axis=0,end_axis=1,target_dims=(self.batch_size,self.num_samp))#[B,N,52,2]

        reward = compute_reward(state_descaled,batch).mean() #[B, num_samp] 
        self.log('val/reward',reward)
        
      



    def on_save_checkpoint(self, checkpoint):
        full_sd = super().state_dict()

        dm_only_sd = {}
        for k, v in full_sd.items():
            if k.startswith("dm."):
                dm_only_sd[k] = v
    
        checkpoint["state_dict"] = dm_only_sd
    # def on_after_backward(self):
    #     max_norm=1e6
    #     total_norm = torch.nn.utils.clip_grad_norm(self.dm.parameters(),max_norm=max_norm)

    #     total_norm=  float(total_norm)
    #     self.log('grad_norm', total_norm,on_step=True,on_epoch=False)          

