from tbsim.models.diffuser import DiffuserModel

import torch.nn.functional as F
import  torch
import matplotlib.pyplot as plt
class DM(DiffuserModel):
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
        diffuser_norm_info,
    ):
        super(DM, self).__init__(map_encoder_model_arch,
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

   