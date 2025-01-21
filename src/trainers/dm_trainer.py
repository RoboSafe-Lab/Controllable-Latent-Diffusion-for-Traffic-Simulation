
import torch.optim as optim
import torch,copy
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from tbsim.models.diffuser_helpers import EMA
from models.vae.vae_model import VaeModel
from models.dm.dm_model import DmModel
class DMLightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes,vae_model_path=None):

        super(DMLightningModule, self).__init__()
        self.algo_config = algo_config
        self.batch_size = train_config.training.batch_size
        self.vae = VaeModel(algo_config,train_config, modality_shapes)
        self.dm = DmModel(
            algo_config.vae.latent_size,
            algo_config.cond_feat_dim,
            algo_config.time_dim,
            algo_config.mlpres_hidden,
            algo_config.mlp_blocks,
            )
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.vae)
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()
        else:
            self.ema_policy=None

        if vae_model_path is not None:
            self._load_vae_weights(vae_model_path)
    

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
        if self.use_ema and "ema_state" in checkpoint:
            ema_dict = checkpoint["ema_state"] 
            for k, v in ema_dict.items():
                if k in self.ema_policy.state_dict():
                    self.ema_policy.state_dict()[k].copy_(v)
                else:
                    pass

            print("Load EMA weights done from ckpt['ema_state'].")
        else:
            print("No 'ema_state' found in ckpt or self.use_ema=False. Skipping EMA load.")

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
        batch = batch_utils().parse_batch(batch)      #TODO:加上self.vae改成ema_policy           
        
        aux_info,unscaled_input,scaled_input = self.ema_policy.pre_vae(batch)
        z = self.ema_policy.lstmvae.getZ(scaled_input,aux_info["cond_feat"])#[B,128]
        loss = self.dm.compute_losses(z,aux_info)

        self.log('train/dm_loss',loss, on_step=True, on_epoch=False,batch_size=self.batch_size)
        return loss
     
  
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch)
       
        aux_info,_,scaled_input = self.ema_policy.pre_vae(batch)
        z = self.ema_policy.lstmvae.getZ(scaled_input,aux_info["cond_feat"])#[B,128]
        loss = self.dm.compute_losses(z,aux_info)

        self.log('val/loss',loss,on_step=False, on_epoch=True,batch_size=self.batch_size)



    # def test_step(self,batch):
    #     num_samp=5
    #     batch = batch_utils().parse_batch(batch)  
    #     aux_info, unscaled_input, scaled_input = self.pre_vae(batch)

    #     z = self.vae.getZ(scaled_input, aux_info["cond_feat"])
    #     sampled_output = self.dm(z, aux_info,num_samp=num_samp) 
    #     traj = self.vae.getTraj(sampled_output,num_samp)
    #     sampled_output_3d = traj.reshape(128, num_samp, 52,2)
    #     all_recon_trajs = []
    #     for k in range(num_samp):
    #         traj_k = sampled_output_3d[:, k, :]
    #         scaled_output_k = self.convert_action_to_state_and_action(traj_k, aux_info['curr_states']) 
    #         descaled_output_k = self.descale_traj(scaled_output_k)  # => shape [B, ...]

    #         all_recon_trajs.append(descaled_output_k)
     
    #     print("1111")
        
    #     custom_dir = "/home/visier/hazardforge/HazardForge/logs/2025-1-9"
    #     for k, recon_traj in enumerate(all_recon_trajs):
    #         origin_traj = unscaled_input.detach().cpu().numpy()

    #         recon_traj = recon_traj.detach().cpu().numpy()
    #         raster_from_agent = batch['raster_from_agent'].detach().cpu().numpy()
    #         maps = batch['maps'].detach().cpu().numpy()
    #         fig = vis_in_out(maps, origin_traj, recon_traj,raster_from_agent, indices=[5,6,7,23,32,90])
            
    #         save_path = os.path.join(custom_dir, f"trajectory_fig_step{self.global_step}_sample{k}.png")

    #         fig.savefig(save_path, dpi=300)

    #     print("1111")




    
    
      
    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.vae.state_dict())

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.vae)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            ema_state = {}
            with torch.no_grad():
                for name,param in self.ema_policy.named_parameters():
                    ema_state[name]=param.detach().cpu().clone()
            checkpoint["ema_state"] = ema_state

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and ("ema_state" in checkpoint):
            ema_state = checkpoint["ema_state"]
            with torch.no_grad():
                for name, param in self.ema_policy.named_parameters():
                    if name in ema_state:
                        param.copy_(ema_state[name])

