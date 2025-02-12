
import torch.optim as optim
import torch,copy
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from tbsim.models.diffuser_helpers import EMA
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
from tbsim.utils.trajdata_utils import get_stationary_mask
from models.rl.criticmodel import compute_reward
class GuideDMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config,modality_shapes):

        super(GuideDMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        
        self.moving_speed_th = algo_config.moving_speed_th
        self.num_samp = algo_config.num_samp
   
        self.dm = DmModel(algo_config,modality_shapes)
       

        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            
            self.ema_dm = copy.deepcopy(self.dm)
            self.ema_dm.requires_grad_(False)

            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()
        else:
            self.ema_policy=None
        
        
    def configure_optimizers(self):  
        optim_params_dm = self.algo_config.optim_params["dm"]
        optimizer = optim.Adam(
            params=self.dm.parameters(),
            lr=optim_params_dm["learning_rate"]["initial"],
            weight_decay=optim_params_dm["regularization"]["L2"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,                  
            steps_per_epoch=7000,         
            epochs=10,                    
            pct_start=0.3,                
            anneal_strategy='cos',        
            div_factor=25,                
            final_div_factor=1000         
        )
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                                },
                }
            
  
    def training_step(self, batch):
        batch = batch_utils().parse_batch(batch) 
        out_dict,image = self(batch,self.num_samp)
        reward = compute_reward(out_dict['pred_traj'],batch)

        diffusion = out_dict['diffusion']#[B,5,52,101,6]
        model_means = out_dict['mean']  #[B,5,52,100,2]         
        log_vars = out_dict['log_var'] #[B,5,1,100,1]
        
        x_sample = diffusion[...,1:, -2:]

        std = (0.5*log_vars).exp()
        dist = torch.distributions.Normal(model_means,std)
        log_prob = dist.log_prob(x_sample)#[B,5,52,100,2]
        
        
        total_log_prob = log_prob.sum(dim=(-1,-2,-3))#[B,5]
        total_log_prob_mean = total_log_prob.mean(dim=1)
        loss = -(reward)* total_log_prob_mean
        
        loss = loss.mean()
        self.log("train/loss", loss, on_step=True, logger=True)
        return loss
    
    def forward(self,obs_dict,num_samp):
        cur_policy = self.dm
      
        self.stationary_mask = get_stationary_mask(obs_dict,self.disable_control_on_stationary,self.moving_speed_th)#[B]
        B = self.stationary_mask.shape[0]
        stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B,num_samp).reshape(B*num_samp)#[B*N]
        return cur_policy(obs_dict,stationary_mask_expand,self.algo_config)
    

        


  
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch) 
        out_dict,image = self(batch,self.num_samp)
        reward = compute_reward(out_dict['pred_traj'],batch)

        diffusion = out_dict['diffusion']#[B,5,52,101,6]
        model_means = out_dict['mean']  #[B,5,52,100,2]         
        log_vars = out_dict['log_var'] #[B,5,1,100,1]
        
        x_sample = diffusion[...,1:, -2:]

        std = (0.5*log_vars).exp()
        dist = torch.distributions.Normal(model_means,std)
        log_prob = dist.log_prob(x_sample)#[B,5,52,100,2]
        
        
        total_log_prob = log_prob.sum(dim=(-1,-2,-3))#[B,5]
        total_log_prob_mean = total_log_prob.mean(dim=1)
        loss = -(reward)* total_log_prob_mean
        
        loss = loss.mean()
        self.log("val/loss", loss, on_step=True, logger=True)
      

    def reset_parameters(self):
        self.ema_dm.load_state_dict(self.dm.state_dict())

    def on_after_optimizer_step(self, optimizer, optimizer_idx):
        if self.use_ema and (self.global_step % self.ema_update_every == 0):
            self.step_ema(self.global_step)

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_dm, self.dm)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            ema_state = {}
            with torch.no_grad():
                for name,param in self.ema_dm.named_parameters():
                    ema_state[name]=param.detach().cpu().clone()
            checkpoint["ema_state_dm"] = ema_state

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and ("ema_state_dm" in checkpoint):
            ema_state = checkpoint["ema_state_dm"]
            with torch.no_grad():
                for name, param in self.ema_dm.named_parameters():
                    if name in ema_state:
                        param.copy_(ema_state[name])
            
                    
              

      # def state_dict(self, *args, **kwargs):
    #     sd = super().state_dict(*args, **kwargs)
    #     sd = {k: v for k, v in sd.items() if not k.startswith("old_dm.")}
    #     return sd
