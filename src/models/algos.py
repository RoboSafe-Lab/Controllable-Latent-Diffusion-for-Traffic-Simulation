from tbsim.algos.algos import DiffuserTrafficModel
import torch.optim as optim
import torch
import pytorch_lightning as pl
from models.dm_vae import  DMVAE
class Diffusion_Decoder(DiffuserTrafficModel):
    def __init__(self, algo_config, modality_shapes, registered_name,
                 do_log=True, guidance_config=None, constraint_config=None):

        super(Diffusion_Decoder, self).__init__(algo_config, modality_shapes,
                                                registered_name, do_log=do_log,
                                                guidance_config=guidance_config,
                                                constraint_config=constraint_config)


        self.validation_epoch_end = lambda *args, **kwargs: None
        self.training_epoch_end = lambda *args, **kwargs: None
        if algo_config.diffuser_input_mode == 'state':
            observation_dim = 0
            action_dim = 3  # x, y, yaw
            output_dim = 3  # x, y, yaw
        elif algo_config.diffuser_input_mode == 'action':
            observation_dim = 0
            action_dim = 2  # acc, yawvel
            output_dim = 2  # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4  # x, y, vel, yaw
            action_dim = 2  # acc, yawvel

            output_dim = 2  # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action_no_dyn':
            observation_dim = 4  # x, y, vel, yaw
            action_dim = 2  # acc, yawvel
            output_dim = 6  # x, y, vel, yaw, acc, yawvel
        else:
            raise

        print(f"outputdim={observation_dim}, actiondim={action_dim}, outputdim={output_dim}")


        diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
        agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
        neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']

        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p]) #note: need to know what is this
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        self.nets["dm_vae"] = DMVAE(
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,

            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames + 1,  # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,

            cond_feature_dim=algo_config.cond_feat_dim,
            curr_state_feature_dim=algo_config.curr_state_feat_dim,

            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,

            observation_dim=observation_dim,
            action_dim=action_dim,

            output_dim=output_dim,

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
            use_reconstructed_state=algo_config.use_reconstructed_state,

            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,

            diffuser_norm_info=diffuser_norm_info,
            agent_hist_norm_info=agent_hist_norm_info,
            neighbor_hist_norm_info=neighbor_hist_norm_info,

            disable_control_on_stationary=self.disable_control_on_stationary,

        )


    def validation_epoch_end(self, *args, **kwargs):
        pass

    def training_epoch_end(self, *args, **kwargs):
        pass

    def forward_vae_decoder(self, x):
        pass
    def forward_dm(self,x):
        pass
    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        pass

    def set_guidance(self, guidance_config, example_batch=None):
        pass

    def on_train_batch_start(self, batch, logs=None):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        print("training_step_start--------------------------------------------------")
        dm_output, vae_output = self.nets['dm_vae'](batch)

        if optimizer_idx == 0:  # 针对 DM 的优化器
            loss_dm = self.compute_dm_loss(dm_output, batch)
            self.log("dm_loss", loss_dm)
            return loss_dm
        elif optimizer_idx == 1:  # 针对 VAE 的优化器
            loss_vae = self.compute_vae_loss(vae_output, batch)
            self.log("vae_loss", loss_vae)
            return loss_vae



    def compute_dm_loss(self, dm_output, batch):
        pass
    def compute_vae_loss(self, vae_output, batch):
        pass


    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        print("Optimizer 1 state:", self.optimizers()[0].state_dict())
        print("Optimizer 2 state:", self.optimizers()[1].state_dict())






    def configure_optimizers(self):
        optim_params_dm = self.algo_config.optim_params["dm"]
        optim_params_vae = self.algo_config.optim_params["vae"]

        optimizer_dm = optim.Adam(
            params=self.nets["dm_vae"].dm.parameters(),
            lr=optim_params_dm["learning_rate"]["initial"],
            weight_decay=optim_params_dm["regularization"]["L2"],
        )
        optimizer_vae = optim.Adam(
            params=self.nets["dm_vae"].vae.parameters(),
            lr=optim_params_vae["learning_rate"]["initial"],
            weight_decay=optim_params_vae["regularization"]["L2"],
        )

        scheduler_dm = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_dm,
            mode='min',
            factor=self.algo_config.optim_params["dm"]["learning_rate"]["decay_factor"],
            patience=5,
            verbose=True
        )
        scheduler_vae = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_vae,
            milestones=optim_params_vae["learning_rate"]["epoch_schedule"],
            gamma=optim_params_vae["learning_rate"]["decay_factor"],
        )

        return (
            [optimizer_dm, optimizer_vae],
            [{"scheduler": scheduler_dm, "interval": "epoch","monitor": "val_loss"},
             {"scheduler": scheduler_vae, "interval": "epoch"}],
        )






