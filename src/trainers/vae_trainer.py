import torch.optim as optim
import torch,copy
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
from tbsim.models.diffuser_helpers import EMA
import os
from models.vae.vae_model import VaeModel

class VAELightningModule(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes):

        super(VAELightningModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
      
        print(f"algo_config_diffuser_input_mode: {algo_config.diffuser_input_mode}")

        self.vae = VaeModel(algo_config,train_config, modality_shapes)
        self.batch_size = train_config.training.batch_size
        '''
        (B,256)->(B,1,256)

        (B,52,x,y,vel,yaw,)
            +
        map:(B,seq,256)考虑current_state 目前先这样做,以后去掉也方便
        '''
        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.vae)
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.beta = 0.01
        self.beta_max = 0.1
        self.anneal_steps = self.train_config.training.num_steps/3

        self.beta_inc = (self.beta_max - self.beta) / self.anneal_steps
        self.val_batch_size = self.train_config.validation.batch_size
   
    def configure_optimizers(self):
        optim_params_vae = self.algo_config.optim_params["vae"]
        optimizer = optim.Adam(
            params=self.vae.parameters(),
            lr=optim_params_vae["learning_rate"]["initial"],
            weight_decay=optim_params_vae["regularization"]["L2"],
            
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)
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

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
  
    def training_step(self, batch):         
        batch = batch_utils().parse_batch(batch)                 
        outputs,losses = self.vae(batch,self.beta)
       
        self.log("train/kl_loss", losses["KLD"],                       on_step=True, on_epoch=False,batch_size=self.batch_size)
        self.log("train/recon_loss", losses["Reconstruction_Loss"],    on_step=True, on_epoch=False,batch_size=self.batch_size)
        self.log("train/total_loss", losses["loss"],                   on_step=True, on_epoch=False,batch_size=self.batch_size)
        outputs["loss"]=losses["loss"]
        return outputs
        
      
    def validation_step(self, batch):
        batch = batch_utils().parse_batch(batch)
        _,losses = self.vae(batch,self.beta)
        self.log("val/loss", losses["loss"],                 on_step=False, on_epoch=True,batch_size=self.batch_size)
        

        

    def on_after_optimizer_step(self, optimizer, optimizer_idx):
        if self.use_ema and (self.global_step % self.ema_update_every == 0):
            self.step_ema(self.global_step)
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


 
    def on_train_batch_end(self, outputs, batch, batch_idx):
      
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        if self.beta < self.beta_max:
            self.beta += self.beta_inc
            self.beta = min(self.beta, self.beta_max)
        self.log("lr", current_lr, on_step=True, on_epoch=False)
        self.log("beta", self.beta, on_step=True, on_epoch=False)

        # if (self.global_step % self.plot_interval == 0) and (self.global_step > 0):
            
        #     origin_traj = outputs['input'].detach().cpu().numpy()
        #     recon_traj = outputs['output'].detach().cpu().numpy()
        #     raster_from_agent = outputs['raster_from_agent'].detach().cpu().numpy()
        #     maps = outputs['maps'].detach().cpu().numpy()
        #     fig = vis_in_out(maps, origin_traj, recon_traj,raster_from_agent, indices=[0,1,2,3,10,20,21,22,23,24,30,80,90,100])

        #     save_path = os.path.join(self.image_dir, f"trajectory_fig_step{self.global_step}.png")
        #     fig.savefig(save_path, dpi=300)

        #     # Close the figure to free memory
        #     plt.close(fig)

  
        
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



