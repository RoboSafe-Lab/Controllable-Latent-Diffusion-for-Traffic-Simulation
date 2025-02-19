
import torch.optim as optim
import torch,copy
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
import math
from models.rl.criticmodel import compute_reward,ReplayBuffer,detach_aux_info
from torch.optim.lr_scheduler import LambdaLR
import tbsim.utils.tensor_utils as TensorUtils
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

        x1 = out_dict['x1']         # shape: [B* num_samp, horizon, latent_size]
        x0 = out_dict['pred_traj']  # shape: [B* num_samp, horizon, latent_size]
        

        aux_info = out_dict['aux_info']
        # aux_info_detached = detach_aux_info(aux_info)

        action_decoder = self.vae.lstmvae.lstm_dec(x0,aux_info['cond_feat'])#[B*N,52,2]
        recon_state_and_action_descaled = self.vae.convert_action_to_state_and_action(action_decoder,aux_info['curr_states'],descaled_output=True)
      
        state_descaled = recon_state_and_action_descaled[...,:2] #[B*N,52,2]
        state_descaled = TensorUtils.reshape_dimensions(state_descaled,begin_axis=0,end_axis=1,target_dims=(self.batch_size,self.num_samp))#[B,N,52,2]

        reward = compute_reward(state_descaled,batch) #[B, num_samp]
        baseline = reward.mean(dim=1, keepdim=True)  # [B,1]
        advantage = reward - baseline  # [B, num_samp]

        log_prob_old = self.dm.log_prob(x1, x0, aux_info,t=torch.zeros(x0.shape[0], device=x0.device, dtype=torch.long)) #[B*N,52,4]
       
        aux_info_detached = detach_aux_info(aux_info)
        self.replay_buffer.add(x0,x1,log_prob_old,advantage,aux_info_detached)
        self.steps_since_update += 1
       
        vis_dict={
            "raster_from_agent":batch['raster_from_agent'],
            "image":batch['image'],
            'traj':state_descaled,
        }
        print(self.global_step)
        if len(self.replay_buffer) < self.buffer_max or self.steps_since_update < self.update_interval:
            self.log('train/reward', reward.mean())
            
            return vis_dict
        
        else:
            ppo_loss = self.ppo_update()
            self.log('train/ppo_loss', ppo_loss)
            self.log('train/reward', reward.mean())
            self.steps_since_update = 0
            return vis_dict
    def ppo_update(self):
        
        eps = 0.2
        losses = []
        for _ in range(self.ppo_update_times):
            batch = self.replay_buffer.sample(self.ppo_sample)
            x0_batch, x1_batch, log_prob_old_batch, advantage, aux_info_batch= zip(*batch)
            x0_batch = torch.stack(x0_batch).to(self.device)
            x1_batch = torch.stack(x1_batch).to(self.device)

            x0_batch = x0_batch.view(-1, x0_batch.size(-2), x0_batch.size(-1)) #[M * (B*num_samp), 52, 4]
            x1_batch = x1_batch.view(-1, x1_batch.size(-2), x1_batch.size(-1))

            log_prob_old_batch = torch.stack(log_prob_old_batch).to(self.device)
            log_prob_old_batch = log_prob_old_batch.view(-1)

            advantage = torch.stack(advantage).to(self.device)
            advantage=advantage.view(-1)

            # aux_info = torch.stack(aux_info).to(self.device)
            # aux_info = aux_info.view(-1,aux_info.size(-1))

            cond_feat_list = [d['cond_feat'] for d in aux_info_batch]
            aux_info_stacked = torch.stack(cond_feat_list, dim=0).view(-1, cond_feat_list[0].shape[-1]).to(self.device)

            log_prob_new = self.dm.log_prob(x1_batch, x0_batch, {'cond_feat':aux_info_stacked}, t=torch.zeros(x0_batch.shape[0], device=self.device, dtype=torch.long))#[M*B*N]
            ratios = torch.exp(log_prob_new - log_prob_old_batch)

            # advantage = advantage - advantage.mean(dim=1, keepdim=True)#NOTE:根据实际计算情况看要不要再次归一化
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantage
            loss = -torch.min(surr1, surr2).mean()
            losses.append(loss)

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        return torch.stack(losses).mean()
      
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

