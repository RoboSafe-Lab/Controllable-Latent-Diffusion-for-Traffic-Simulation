import pytorch_lightning as pl
import torch
from models.vae.transformer_vae import TransformerVAE

class VAETrainer(pl.LightningModule):
    def __init__(self, vae_config, training_config):
        super().__init__()
        self.model = TransformerVAE(**vae_config)
        self.training_config = training_config
        self.save_hyperparameters()
        
    def forward(self, x):   
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        recon, mean, logvar = self.model(batch)
        loss = self._compute_loss(recon, batch, mean, logvar)
        self.log('train_loss', loss)
        return loss

    def _compute_loss(self, recon, x, mean, logvar):
        # VAE Loss (Reconstruction + KL)
        recon_loss = torch.nn.functional.mse_loss(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + self.hparams.training_config["kl_weight"] * kl_loss
    
    def validation_step(self, batch, batch_idx):
        recon, mean, logvar = self.model(batch)
        loss = self._compute_loss(recon, batch, mean, logvar)
        self.log('val_loss', loss)
        
        # 计算并记录单独的重构损失和KL损失
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        recon, mean, logvar = self.model(batch)
        loss = self._compute_loss(recon, batch, mean, logvar)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config.get("weight_decay", 0.0)
        )
        
        # 简单的学习率调度器
        if "lr_scheduler" in self.training_config:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        
        return optimizer

    
    def on_train_epoch_end(self):
        """训练轮结束时的回调"""
        # 记录当前学习率
        for param_group in self.trainer.optimizers[0].param_groups:
            current_lr = param_group['lr']
            self.log('learning_rate', current_lr)