import torch.optim as optim
import torch
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from models.vae.vae_model import VaeModel
from torch.optim.lr_scheduler import LambdaLR
import math
class VAELightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes):

        super(VAELightningModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.epochs = train_config.training.epochs
      

        self.vae = VaeModel(algo_config,train_config, modality_shapes)
        self.batch_size = train_config.training.batch_size
        
        self.beta = 0.05
        self.beta_max = 0.3
        

        self.beta_inc = (self.beta_max - self.beta) / 9000
        self.val_batch_size = self.train_config.validation.batch_size
   
    def configure_optimizers(self):
        optim_params_vae = self.algo_config.optim_params["vae"]
        optimizer = optim.Adam(
            params=self.vae.parameters(),
            lr=optim_params_vae["learning_rate"]["initial"],
            weight_decay=optim_params_vae["regularization"]["L2"], 
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
        output_dict = self.vae(batch,self.beta)
       
        self.log("train/kld", output_dict["kld"],        on_step=True, on_epoch=False,batch_size=self.batch_size)
        self.log("train/recon", output_dict["recon"],    on_step=True, on_epoch=False,batch_size=self.batch_size)
        self.log("train/vae_loss", output_dict["loss"],  on_step=True, on_epoch=False,batch_size=self.batch_size)

        self.log("train/beta",self.beta,on_step=True, on_epoch=False,batch_size=self.batch_size)
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



