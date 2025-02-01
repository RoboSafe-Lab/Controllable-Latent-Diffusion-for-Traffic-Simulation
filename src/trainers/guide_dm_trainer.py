
import torch.optim as optim
import torch,copy
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from tbsim.models.diffuser_helpers import EMA
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
from tbsim.utils.trajdata_utils import get_stationary_mask
class GuideDMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes,vae_model_path):

        super(GuideDMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary



        self.vae = VaeModel(algo_config,train_config, modality_shapes)
        self.dm = DmModel(
            algo_config.vae.latent_size,
            algo_config.cond_feat_dim,
            algo_config.time_dim,
            algo_config.mlp_blocks,
            algo_config.num_infer,
            )
        self.rl = None
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            
            self.ema_vae = copy.deepcopy(self.vae) 
            self.ema_vae.requires_grad_(False)

            self.ema_dm = copy.deepcopy(self.dm)
            self.ema_dm.requires_grad_(False)

            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()
        else:
            self.ema_policy=None

        if vae_model_path is not None:
            self._load_vae_weights(vae_model_path)
        
        self.old_dm = copy.deepcopy(self.dm)
        
    def _load_vae_weights(self, ckpt_path: str):
       
        print(f"Loading VAE weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        lightning_sd = checkpoint["state_dict"] 
        new_sd = {}
        for old_key, val in lightning_sd.items():
            if old_key.startswith("vae."):
                new_key = old_key[len("vae."):] 
                new_sd[new_key] = val
            else:
                pass
        missing, unexpected = self.vae.load_state_dict(new_sd, strict=False)
        print("Load normal VAE weights done. missing:", missing, "unexpected:", unexpected)
        for param in self.vae.parameters():
                param.requires_grad = False

        if self.use_ema and ("ema_state" in checkpoint):
            ema_state = checkpoint["ema_state"]
            with torch.no_grad():
                for name, param in self.ema_vae.named_parameters():
                    if name in ema_state:
                        param.copy_(ema_state[name])
    '''
    def _load_weights(self, dm,rl: str,):
       
        if dm is not None:
            print(f"Loading from: {dm}")
            checkpoint = torch.load(dm, map_location="cpu")
            state_dict = checkpoint["state_dict"]

        vae_sd = {}
        for old_key, val in state_dict.items():
                if old_key.startswith("vae."):
                    new_key = old_key[len("vae."):]
                    vae_sd[new_key] = val
        missing_vae, unexpected_vae = self.vae.load_state_dict(vae_sd, strict=False)
        print("Loaded VAE from ckpt. missing:", missing_vae, "unexpected:", unexpected_vae)
        for param in self.vae.parameters():
                param.requires_grad = False

        dm_sd = {}
        for old_key, val in state_dict.items():
            if old_key.startswith("dm."):
                new_key = old_key[len("dm."):]
                dm_sd[new_key] = val
        missing_dm, unexpected_dm = self.dm.load_state_dict(dm_sd, strict=False)
        print("Loaded DM from ckpt. missing:", missing_dm, "unexpected:", unexpected_dm)

        if self.use_ema and "ema_state" in checkpoint:
                ema_dict = checkpoint["ema_state"]
                for k, v in ema_dict.items():
                    # 注意这里: 只要我们在ema_policy里找得到同名参数，就copy
                    if k in self.ema_policy.state_dict():
                        self.ema_policy.state_dict()[k].copy_(v)
                        print("Loaded EMA weights from ckpt['ema_state'] into ema_policy.")
                    elif self.use_ema:
                        print("No 'ema_state' found in ckpt or use_ema=False. Skipping EMA load.")

        if rl is not None:
            # TODO: Load RL Model Weights
            pass
    '''

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

        aux_info,*_ = self.ema_vae.pre_vae(batch)
        with torch.no_grad():
            x0_old, traj_data_old = self.old_dm(batch, aux_info, self.algo_config)
        trajectory = self.ema_vae.z2traj(x0_old,aux_info)#翻译到物理空间
        

        logp_new_list = []
        logp_old_list = []
        for step_info in traj_data_old:
            x_t = step_info["x_t"]
            x_tminus1 = step_info["x_tminus1"]
            old_mean_t = step_info["mean_t"]
            old_sigma_t = step_info["sigma_t"]
            t = step_info["t"]

            new_model_mean, _, model_log_variance = self.dm.x_tminus1_mean(x_t,t,aux_info)
            new_sigma = (0.5 * model_log_variance).exp()
            dist_new = torch.distributions.Normal(new_model_mean, new_sigma)
            logp_new_t = dist_new.log_prob(x_tminus1).sum(dim=-1)  # [B]


            dist_old = torch.distributions.Normal(old_mean_t, old_sigma_t)
            logp_old_t = dist_old.log_prob(x_tminus1).sum(dim=-1)  # [B]

            logp_new_list.append(logp_new_t)
            logp_old_list.append(logp_old_t)

        logp_new_stacked = torch.stack(logp_new_list, dim=0)
        logp_old_stacked = torch.stack(logp_old_list, dim=0)  

        ratio_stacked = torch.exp(logp_new_stacked - logp_old_stacked)  # [T, B]
        ratio_sum = ratio_stacked.sum(dim=0)  # => [B]
        
        reward = self.rl(batch, trajectory, aux_info) 
        loss = -(ratio_sum * reward).mean() 
        
        self.log("train/loss", loss, on_step=True, logger=True)
        return loss


    def on_after_backward(self):
        if (self.global_step % 10) == 0:
            self.old_dm.load_state_dict(self.dm.state_dict())
  
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch)
       
        aux_info,_,scaled_input = self.ema_policy.pre_vae(batch)
        z = self.ema_policy.lstmvae.getZ(scaled_input,aux_info["cond_feat"])#[B,128]
        loss = self.dm.compute_losses(z,aux_info)

        self.log('val/loss',loss,on_step=False, on_epoch=True,batch_size=self.batch_size)






    
    
      
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
        if self.use_ema and ("ema_state" in checkpoint):
            ema_state = checkpoint["ema_state_dm"]
            with torch.no_grad():
                for name, param in self.ema_dm.named_parameters():
                    if name in ema_state:
                        param.copy_(ema_state[name])

