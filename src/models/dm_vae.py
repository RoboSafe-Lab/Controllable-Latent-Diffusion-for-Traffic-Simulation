from tbsim.models.diffuser import DiffuserModel
from  models.vae import HF_CVAE
import torch.nn.functional as F
import  torch
import matplotlib.pyplot as plt
class DMVAE(DiffuserModel):
    def __init__(
        self,
        map_encoder_model_arch,
        input_image_shape,
        map_feature_dim,
        map_grid_feature_dim,
        diffuser_model_arch,
        horizon,
        observation_dim,
        action_dim,
        output_dim,
        cond_feature_dim,
        curr_state_feature_dim,
        rasterized_map,
        use_map_feat_global,
        use_map_feat_grid,
        rasterized_hist,
        hist_num_frames,
        hist_feature_dim,
        n_timesteps,
        loss_type,
        clip_denoised,
        predict_epsilon,
        action_weight,
        loss_discount,
        loss_weights,
        dim_mults,
        dynamics_type,
        dynamics_kwargs,
        base_dim,
        diffuser_building_block,
        action_loss_only,
        diffuser_input_mode,
        use_conditioning,
        cond_fill_value,
        disable_control_on_stationary,

        trajectory_shape: tuple,  # [T, D]
        latent_dim: int,
        step_time: int,
        diffuser_norm_info,
        vae_hidden_dims: tuple = (128, 256, 512),
        vae_loss_weight: float = 1.0,
        dm_loss_weight: float = 1.0,

    ):
        super(DMVAE, self).__init__(map_encoder_model_arch,
                                    input_image_shape,
                                    map_feature_dim,
                                    map_grid_feature_dim,
                                    diffuser_model_arch,
                                    horizon,
                                    observation_dim,
                                    action_dim,
                                    output_dim,
                                    cond_feature_dim,
                                    curr_state_feature_dim,
                                    rasterized_map,
                                    use_map_feat_global,
                                    use_map_feat_grid,
                                    rasterized_hist,
                                    hist_num_frames,
                                    hist_feature_dim,
                                    n_timesteps,
                                    loss_type,
                                    clip_denoised,
                                    predict_epsilon,
                                    action_weight,
                                    loss_discount,
                                    loss_weights,
                                    dim_mults,
                                    dynamics_type,
                                    dynamics_kwargs,
                                    base_dim,
                                    diffuser_building_block,
                                    action_loss_only,
                                    diffuser_input_mode,
                                    use_conditioning,
                                    cond_fill_value,
                                    disable_control_on_stationary,
                                    diffuser_norm_info=diffuser_norm_info,
                                    dt=0.1,
                                    )
        self.vae = HF_CVAE(
            trajectory_shape=trajectory_shape,
            latent_dim=latent_dim,
            mlp_layer_dims=vae_hidden_dims,
            step_time=step_time,
            diffuser_norm_info=diffuser_norm_info,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
        )

        self.vae_loss_weight = vae_loss_weight
        self.dm_loss_weight = dm_loss_weight



        self.vae_loss_weight = vae_loss_weight

    def get_vaeloss(self,data_batch):
        aux_info = self.get_aux_info(data_batch)
        target_traj = self.get_state_and_action_from_data_batch(data_batch)
        x = self.scale_traj(target_traj)#(B,52,6)


        vae_output= self.vae(x,aux_info)
        mu, logvar= vae_output['encoder_output']['mu'],vae_output['encoder_output']['logvar']
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        decoder_x = vae_output['decoder_output']['trajectories']

        recon_loss = F.mse_loss(x, decoder_x,reduction='mean')

        return kl_loss, recon_loss




    def forward(self, x, aux_info, time):
        """
        Forward pass for DMVAE.

        Args:
            x (torch.Tensor): Input features for the diffusion model.
            aux_info (dict): Auxiliary information.
            time (torch.Tensor): Time step for the diffusion model.

        Returns:
            dict: Outputs for DM and VAE.
        """
        # Diffusion model forward
        dm_output = super().forward(x, aux_info, time)

        # Flatten DM output for VAE
        x_flattened = dm_output.view(dm_output.size(0), -1)

        # VAE forward
        outputs = self.vae(inputs={"trajectories": x_flattened}, condition_inputs=aux_info)

        return {
            "dm_output": dm_output,
            "vae_outputs": outputs,
        }

    def compute_dmvae_loss(self, dm_output, vae_outputs, target):
        """
        Compute the combined loss for DM and VAE.

        Args:
            dm_output (torch.Tensor): Output of the diffusion model.
            vae_outputs (dict): Output of the VAE.
            target (torch.Tensor): Ground truth trajectory.

        Returns:
            torch.Tensor: Combined loss.
        """
        # VAE loss
        vae_loss_dict = self.vae.compute_loss(vae_outputs, target)
        vae_loss = vae_loss_dict["total_loss"]

        # DM loss
        dm_loss = F.mse_loss(dm_output, target, reduction="mean")

        # Combined loss
        total_loss = self.vae_loss_weight * vae_loss + self.dm_loss_weight * dm_loss

        return {
            "total_loss": total_loss,
            "vae_loss": vae_loss,
            "dm_loss": dm_loss,
        }


