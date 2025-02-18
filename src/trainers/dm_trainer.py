
import torch.optim as optim
import torch
torch.set_float32_matmul_precision('medium')
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from models.dm.dm_model import DmModel
import torch.nn.functional as F
from tbsim.utils.trajdata_utils import get_stationary_mask
from models.vae.vae_model import VaeModel
from torch.optim.lr_scheduler import LambdaLR
import math
class DMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes,ckpt_dm):

        super(DMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
        self.epochs = train_config.training.epochs
     
        self.num_samp = algo_config.num_samp

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
        aux_info,  batch_state_and_action_scaled = self.vae.pre_vae(batch)

        z0,_,_ = self.vae.lstmvae.traj2z(batch_state_and_action_scaled, aux_info["cond_feat"])#[B,52,4]
        loss = self.dm.compute_losses(aux_info,z0)

        self.log('train/dm_loss',loss, on_step=True, on_epoch=False,batch_size=self.batch_size,prog_bar=True)
        return loss
        
        
    
    def get_action(self,obs_dict,num_action_samples=1):
      pass
 
        
    def forward(self,obs_dict,num_samp):
        
        pass


  
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch)  
        aux_info,  batch_state_and_action_scaled = self.vae.pre_vae(batch)
        z0,_,_ = self.vae.lstmvae.traj2z(batch_state_and_action_scaled, aux_info["cond_feat"])
        loss = self.dm.compute_losses(aux_info,z0)
        
        self.log('val/loss',loss,on_step=False, on_epoch=True,batch_size=self.batch_size)
   
    
      
    def on_save_checkpoint(self, checkpoint):
        full_sd = super().state_dict()

        dm_only_sd = {}
        for k, v in full_sd.items():
            if k.startswith("dm."):
                dm_only_sd[k] = v
    
        checkpoint["state_dict"] = dm_only_sd

    def on_after_backward(self):
        max_norm=1e6
        total_norm = torch.nn.utils.clip_grad_norm(self.dm.parameters(),max_norm=max_norm)

        total_norm=  float(total_norm)
        self.log('grad_norm', total_norm,on_step=True,on_epoch=False)