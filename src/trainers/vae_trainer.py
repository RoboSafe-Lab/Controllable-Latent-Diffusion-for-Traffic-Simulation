import torch.optim as optim
import torch
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from models.vae.vae_model import VaeModel

class VAELightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes):

        super(VAELightningModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
      
      

        self.vae = VaeModel(algo_config,train_config, modality_shapes)
        self.batch_size = train_config.training.batch_size
        
        self.beta = 0.001
        self.beta_max = 0.1
        

        self.beta_inc = (self.beta_max - self.beta) / 3000
        self.val_batch_size = self.train_config.validation.batch_size
   
    def configure_optimizers(self):
        optim_params_vae = self.algo_config.optim_params["vae"]
        optimizer = optim.Adam(
            params=self.vae.parameters(),
            lr=optim_params_vae["learning_rate"]["initial"],
            weight_decay=optim_params_vae["regularization"]["L2"],
            
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            'monitor':'train/vae_loss',
        },
    }


    def training_step(self, batch):         
        batch = batch_utils().parse_batch(batch)     
        output_dict = self.vae(batch,self.beta)
       
        self.log("train/kld", output_dict["kld"],        on_step=True, on_epoch=True,batch_size=self.batch_size)
        self.log("train/recon", output_dict["recon"],    on_step=True, on_epoch=True,batch_size=self.batch_size)
        self.log("train/vae_loss", output_dict["loss"],  on_step=True, on_epoch=True,batch_size=self.batch_size)
      
        return {"loss": output_dict["loss"], "log_dict": output_dict}

        
      
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch)     
        output_dict = self.vae(batch,self.beta)
       
        self.log("val/kld", output_dict["kld"],        on_step=False, on_epoch=True,batch_size=self.batch_size)
        self.log("val/recon", output_dict["recon"],    on_step=False, on_epoch=True,batch_size=self.batch_size)
        self.log("val/loss", output_dict["loss"],      on_step=False, on_epoch=True,batch_size=self.batch_size)
  

    def on_train_batch_end(self, outputs, batch, batch_idx):
    
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        if self.beta < self.beta_max:
            self.beta += self.beta_inc
            self.beta = min(self.beta, self.beta_max)
        self.log("lr", current_lr, on_step=True, on_epoch=False)
        self.log("beta", self.beta, on_step=True, on_epoch=False)    

    # def on_after_optimizer_step(self, optimizer, optimizer_idx):
    #     if self.use_ema and (self.global_step % self.ema_update_every == 0):
    #         self.step_ema(self.global_step)
 

 
 

    # def reset_parameters(self):
    #     self.ema_policy.load_state_dict(self.vae.state_dict())
    # def step_ema(self, step):
    #     if step < self.ema_start_step:
    #         self.reset_parameters()
    #         return
    #     self.ema.update_model_average(self.ema_policy, self.vae)

    # def on_save_checkpoint(self, checkpoint):
    #     if self.use_ema:
    #         ema_state = {}
    #         with torch.no_grad():
    #             for name,param in self.ema_policy.named_parameters():
    #                 ema_state[name]=param.detach().cpu().clone()
    #         checkpoint["ema_state"] = ema_state

    # def on_load_checkpoint(self, checkpoint):
    #     if self.use_ema and ("ema_state" in checkpoint):
    #         ema_state = checkpoint["ema_state"]
    #         with torch.no_grad():
    #             for name, param in self.ema_policy.named_parameters():
    #                 if name in ema_state:
    #                     param.copy_(ema_state[name])



