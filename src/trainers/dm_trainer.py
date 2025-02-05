
import torch.optim as optim
import torch,copy
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from tbsim.models.diffuser_helpers import EMA
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
import torch.nn.functional as F

class DMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes,):

        super(DMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
      
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
        loss,origin,output = self.dm.compute_losses(batch)
        # z = self.ema_vae.lstmvae.getZ(scaled_input,aux_info["context"])#[B,128]
        # z_0_recon = self.dm.compute_losses(z,aux_info)
        # traj_recon = self.ema_vae.z2traj(z_0_recon,aux_info)

        # traj_recon = traj_recon*batch['target_availabilities'].unsqueeze(-1)
        # scaled_input = scaled_input*batch['target_availabilities'].unsqueeze(-1)

        # loss = F.mse_loss(traj_recon,scaled_input)
        self.log('train/dm_loss',loss, on_step=True, on_epoch=False,batch_size=self.batch_size)
        return {"loss":loss,
                "input": origin[...,:2],
                "output":output[...,:2],
                "raster_from_agent":batch['raster_from_agent'],
                "image":batch['image'],

        }
        
     
  
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch)
       
        loss = self.dm.compute_losses(batch)
        # z = self.ema_vae.lstmvae.getZ(scaled_input,aux_info["context"])#[B,128]
        # z_0_recon = self.dm.compute_losses(z,aux_info)
        # traj_recon = self.ema_vae.z2traj(z_0_recon,aux_info)

        # traj_recon = traj_recon*batch['target_availabilities'].unsqueeze(-1)
        # scaled_input = scaled_input*batch['target_availabilities'].unsqueeze(-1)

        # loss = F.mse_loss(traj_recon,scaled_input)

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
