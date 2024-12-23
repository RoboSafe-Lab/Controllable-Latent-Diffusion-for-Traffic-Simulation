from configs.visualize_traj import vis_in_out
from models.vae import LSTMVAE
import torch.optim as optim
import torch,copy
from models.dm import  DM
from tbsim.utils.batch_utils import batch_utils
import pytorch_lightning as pl
import torch.nn as nn
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.models.diffuser_helpers import EMA
import matplotlib.pyplot as plt
import os
import tbsim.dynamics as dynamics
import tbsim.models.base_models as base_models
import numpy as np
from tbsim.models.diffuser_helpers import MapEncoder,convert_state_to_state_and_action,unicyle_forward_dynamics
class UnifiedTrainer(pl.LightningModule):
    def __init__(self, algo_config,train_config, modality_shapes, registered_name,
                 do_log=True, guidance_config=None, constraint_config=None,train_mode="vae", vae_model_path=None):

        super(UnifiedTrainer, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        #self.nets = nn.ModuleDict()
        self._do_log = do_log

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate
        # used only when data_centric == 'scene' and coordinate == 'agent'
        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist
        # to help control stationary agent's behavior
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.stationary_mask = None
        print(f"algo_config_diffuser_input_mode: {algo_config.diffuser_input_mode}")

        
      
        if algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4  # x, y, vel, yaw
            action_dim = 2  # acc, yawvel
            output_dim = 2  # acc, yawvel   
            layer_dims =  (algo_config.curr_state_feat_dim, algo_config.curr_state_feat_dim)  
            self.default_chosen_inds = [0, 1, 2, 3, 4, 5] 
        else:
            raise
        cond_in_feat_size = 0
        cond_in_feat_size += algo_config.curr_state_feat_dim
        self.agent_state_encoder = base_models.MLP(observation_dim,
                                                       algo_config.curr_state_feat_dim,
                                                       layer_dims,
                                                       normalization=True)
        self.map_encoder = MapEncoder(
                model_arch=algo_config.map_encoder_model_arch,
                input_image_shape=modality_shapes["image"],
                global_feature_dim=algo_config.map_feature_dim,
                grid_feature_dim= None,
            )
        cond_in_feat_size += algo_config.map_feature_dim
        cond_out_feat_size = algo_config.cond_feat_dim
        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)
        self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                cond_out_feat_size,
                                                combine_layer_dims,
                                                normalization=True)
        

        diffuser_norm_info = algo_config.nusc_norm_info.diffuser
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32') 


        agent_hist_norm_info = algo_config.nusc_norm_info.agent_hist
        neighbor_hist_norm_info = algo_config.nusc_norm_info.neighbor_hist


        




        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0  # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map
        self.use_rasterized_hist = algo_config.rasterized_history

        if self.use_cond:
            if self.cond_drop_map_p > 0:
                print(
                    'DIFFUSER: Dropping map input conditioning with p = %f during training...' % (self.cond_drop_map_p))
            if self.cond_drop_neighbor_p > 0:
                print('DIFFUSER: Dropping neighbor traj input conditioning with p = %f during training...' % (
                    self.cond_drop_neighbor_p))
                
        self.horizon = algo_config.horizon
        self.dt = 0.1
        '''
        self.dm = DM(
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,
            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            output_dim=output_dim,
            cond_feature_dim=algo_config.cond_feat_dim,
            curr_state_feature_dim=algo_config.curr_state_feat_dim,
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames + 1,  # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,
            n_timesteps=algo_config.n_diffusion_steps,
            loss_type=algo_config.loss_type,
            clip_denoised=algo_config.clip_denoised,
            predict_epsilon=algo_config.predict_epsilon,
            action_weight=algo_config.action_weight,
            loss_discount=algo_config.loss_discount,
            loss_weights=algo_config.diffusor_loss_weights,
            dim_mults=algo_config.dim_mults,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            base_dim=algo_config.base_dim,
            diffuser_building_block=algo_config.diffuser_building_block,
            action_loss_only=algo_config.action_loss_only,
            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,
            disable_control_on_stationary=self.disable_control_on_stationary,
            diffuser_norm_info=diffuser_norm_info,

        )
        
        '''
        
        self._dynamics_type = algo_config.dynamics.type
        self._dynamics_kwargs=algo_config.dynamics
        self._create_dynamics()
        if guidance_config is not None:
            self.set_guidance(guidance_config)
        if constraint_config is not None:
            self.set_constraints(constraint_config)

        
        vae_config = algo_config.vae
        self.vae = LSTMVAE(input_size=observation_dim+action_dim,
                           hidden_size=vae_config.hidden_size,
                           latent_size=vae_config.latent_size,
                           output_size=output_dim,
                           )
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

        self.train_mode = train_mode  # "vae" or "dm"

        if train_mode == "dm":
            # Load pre-trained VAE model and freeze parameters
            if vae_model_path:
                self.vae.load_state_dict(torch.load(vae_model_path))
                for param in self.dm.vae.parameters():
                    param.requires_grad = False

        
     
        self.batch_size = self.train_config.training.batch_size
        self.beta = 0.1
        self.beta_max = 1.0
        self.anneal_steps = self.train_config.training.num_steps/3
        self.beta_inc = (self.beta_max - self.beta) / self.anneal_steps

        self.val_batch_size = self.train_config.validation.batch_size
        self.plot_interval = self.train_config.plt_interval

        '''
        (B,256)->(B,1,256)

        (B,52,x,y,vel,yaw,)
            +
        map:(B,seq,256)考虑current_state 目前先这样做,以后去掉也方便
        '''


    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None
    def configure_optimizers(self):
        
        if self.train_mode == "vae":
            optim_params_vae = self.algo_config.optim_params["vae"]

            all_params = (list(self.vae.parameters())+ 
                          list(self.map_encoder.parameters())+ 
                          list(self.process_cond_mlp.parameters())+ 
                          list(self.agent_state_encoder.parameters()
                        )
)

            optimizer = optim.Adam(
                params=all_params,
                lr=optim_params_vae["learning_rate"]["initial"],
                weight_decay=optim_params_vae["regularization"]["L2"],
              
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

        elif self.train_mode == "dm":
            optim_params_dm = self.algo_config.optim_params["dm"]
            optimizer = optim.Adam(
                params=self.dm.parameters(),
                lr=optim_params_dm["learning_rate"]["initial"],
                weight_decay=optim_params_dm["regularization"]["L2"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=optim_params_dm["learning_rate"]["decay_factor"],
                patience=5,
                verbose=True,
            )
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "dm_loss",
            },
        }
        else:
            raise ValueError(f"Unknown training mode: {self.train_mode}")
        

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
  
    def training_step(self, batch):
        if self.use_ema and self.global_step % self.ema_update_every == 0:
            self.step_ema(self.global_step)
            self.data_centric = 'agent'
        batch = batch_utils().parse_batch(batch)                 
        
        if self.train_mode == "vae":

          
            aux_info,unscaled_input,scaled_input = self.pre_vae(batch)
            scaled_actions,mu,logvar = self.vae(scaled_input,aux_info)
            descaled_output = self.convert_action_to_state_and_action(scaled_actions,aux_info['curr_states'])
            scaled_output = self.scale_traj(descaled_output)

            losses = self.vae.loss_function(scaled_output,scaled_input,mu,logvar,self.beta)
            loss = losses["loss"]
            recon_loss = losses["Reconstruction_Loss"]
            kl_loss = losses["KLD"]
          
            self.log("train/kl_loss", kl_loss,          on_step=True, on_epoch=False,batch_size=self.batch_size)
            self.log("train/recon_loss", recon_loss,    on_step=True, on_epoch=False,batch_size=self.batch_size)
            self.log("train/vae_loss", loss,            on_step=True, on_epoch=False,batch_size=self.batch_size)

          
            return {"loss": loss, 
                    "input": unscaled_input,
                    "output":descaled_output,
                    "raster_from_agent":batch['raster_from_agent'],
                    "maps":batch['maps'],
                    }

        elif self.train_mode == "dm":
            dm_losses_dict = self(batch)
            dm_loss = dm_losses_dict['diffusion_loss']
            self.log("dm_loss", dm_loss,on_step=True,on_epoch=True,batch_size=self.batch_size)
            return dm_loss
     
    def convert_action_to_state_and_action(self, x_out, curr_states, scaled_input=True, descaled_output=False):
        '''
        Apply dynamics on input action trajectory to get state+action trajectory
        Input:
            x_out: (batch_size, num_steps, 2). scaled action trajectory
        Output:
            x_out: (batch_size, num_steps, 6). scaled state+action trajectory
        '''
        dim = len(x_out.shape)
        if dim == 4:
            B, N, T, _ = x_out.shape
            x_out = TensorUtils.join_dimensions(x_out,0,2)

        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel'
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        if dim == 4:
            x_out_all = x_out_all.reshape([B, N, T, -1])
        return x_out_all
    def pre_vae(self,batch):
        aux_info = self.get_aux_info(batch)


        unscaled_input = self.get_state_and_action_from_data_batch(batch)#[B,52,6]
        scaled_input = self.scale_traj(unscaled_input)
        return aux_info,unscaled_input,scaled_input

    def get_aux_info(self,data_batch):
        N = data_batch["history_positions"].size(0)
        device = data_batch["history_positions"].device
        cond_feat_in = torch.empty((N,0)).to(device)

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())#[B,4]
        curr_states_input = self.scale_traj(curr_states,[0,1,2,3])#[B,4]
        curr_state_feat = self.agent_state_encoder(curr_states_input)#[B,64]
        cond_feat_in = torch.cat([cond_feat_in, curr_state_feat], dim=-1)#[B,0+64]

        image_batch = data_batch["image"]#[B,34,224,224]包含历史轨迹和邻居轨迹
        map_global_feat, map_grid_feat = self.map_encoder(image_batch)#[B,256]
        cond_feat_in = torch.cat([cond_feat_in, map_global_feat], dim=-1)#[B,64+256=320]
        cond_feat = self.process_cond_mlp(cond_feat_in)#[B,256]
        aux_info = {
            'cond_feat': cond_feat, #已经归一化输入
            'curr_states': curr_states, #没有归一化
        }
        return aux_info
    
    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        '''
        Extract state and(or) action from the data_batch from data_batch
        Input:
            data_batch: dict
        Output:
            x: (batch_size, num_steps, len(chosen_inds)).
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        # NOTE: for predicted agent, history and future with always be fully available
        traj_state = torch.cat(
                (data_batch["target_positions"][:, :self.horizon, :], data_batch["target_yaws"][:, :self.horizon, :]), dim=2)

        traj_state_and_action = convert_state_to_state_and_action(traj_state, data_batch["curr_speed"], self.dt)

        return traj_state_and_action[..., chosen_inds]
    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        - traj: B x T x D
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds

        squeeze_time_dim = False
        if target_traj_orig.dim() == 2:
        # 变成 [B, 1, D]
            target_traj_orig = target_traj_orig.unsqueeze(1)#[B,1,4]
            squeeze_time_dim = True


        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D     #[1,1,4]
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D     #[1,1,4]

        # TODO make these a buffer so they're put on the device automatically
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div
        if squeeze_time_dim:
            target_traj = target_traj.squeeze(1) 
        return target_traj 

    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        - traj: B x T x D
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add
        

        return target_traj

    def validation_step(self, batch):
        if self.use_ema and self.global_step % self.ema_update_every == 0:
            self.step_ema(self.global_step)
            self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)
        if self.train_mode == "vae":
            aux_info, scaled_traj,scaled_input = self.pre_vae(batch)
            scaled_actions,mu,logvar = self.vae(scaled_traj,aux_info)
            descaled_output = self.convert_action_to_state_and_action(scaled_actions,aux_info['curr_states'])
            scaled_output = self.scale_traj(descaled_output)

            losses = self.vae.loss_function(scaled_output,scaled_input,mu,logvar,self.beta)
            validation_loss = losses["loss"]
            recons_loss = losses["Reconstruction_Loss"]
            kld_loss = losses["KLD"] 

            self.log("val/loss", validation_loss,      on_step=False, on_epoch=True,batch_size=self.batch_size)
            self.log("val/decoder_loss", recons_loss,  on_step=False, on_epoch=True,batch_size=self.batch_size)
            self.log("val/kl", kld_loss,               on_step=False, on_epoch=True,batch_size=self.batch_size)




        

        elif self.train_mode == "dm":
           pass
            

 
    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        if self.beta < self.beta_max:
            self.beta += self.beta_inc
            self.beta = min(self.beta, self.beta_max)
        self.log("lr", current_lr, on_step=True, on_epoch=False)
        self.log("beta", self.beta, on_step=True, on_epoch=False)

        if (self.global_step % self.plot_interval == 0) and (self.global_step > 0):
            
            origin_traj = outputs['input'].detach().cpu().numpy()
            recon_traj = outputs['output'].detach().cpu().numpy()
            raster_from_agent = outputs['raster_from_agent'].detach().cpu().numpy()
            maps = outputs['maps'].detach().cpu().numpy()
            fig = vis_in_out(maps, origin_traj, recon_traj,raster_from_agent, indices=[5, 50, 100, 111])

            save_path = os.path.join(self.image_dir, f"trajectory_fig_step{self.global_step}.png")
            fig.savefig(save_path, dpi=300)

            # Close the figure to free memory
            plt.close(fig)













 

    def on_train_epoch_end(self):
        print(f"-----Epoch {self.current_epoch} has ended. Total Steps: {self.global_step}")


    def compute_dm_loss(self, dm_output, batch):
       
        pass

    @property
    def checkpoint_monitor_keys(self):
    

        if self.train_mode == "vae":
            # Monitor metrics specific to VAE training
            return {"val_loss": "val/loss"}
        elif self.train_mode == "dm":
            return {"val_loss": "val/loss"}
        else:
            raise ValueError(f"Unknown train mode: {self.train_mode}")

  
    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.vae.state_dict())
    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.vae)



