from tbsim.models.diffuser import DiffuserModel
from  models.vae import HF_CVAE
import torch.nn.functional as F
import  torch
class DMVAE(DiffuserModel):
    def __init__(
        self,
        trajectory_shape: tuple,  # [T, D]
        condition_dim: int,
        latent_dim: int,
        step_time: int,
        dynamics_type: str = None,
        dynamics_kwargs: dict = None,
        vae_hidden_dims: tuple = (128, 256, 512),
        vae_loss_weight: float = 1.0,
        dm_loss_weight: float = 1.0,
        **kwargs
    ):
        super(DMVAE, self).__init__(**kwargs)
        self.vae = HF_CVAE(
            trajectory_shape=trajectory_shape,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            mlp_layer_dims=vae_hidden_dims,
            step_time=step_time,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
        )
        self.vae_loss_weight = vae_loss_weight
        self.dm_loss_weight = dm_loss_weight



        self.vae_loss_weight = vae_loss_weight

    def get_vaeloss(self,data_batch):
        aux_info = self.get_aux_info(data_batch)
        target_traj = self.get_state_and_action_from_data_batch(data_batch)
        if self.use_reconstructed_state and self.diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:#TODO:目前没有执行
            target_traj = self.convert_action_to_state_and_action(target_traj[..., [4, 5]], aux_info['curr_states'], scaled_input=False)

        x = self.scale_traj(target_traj)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()



        x = self.scale_traj(target_traj)
        x_start = x
        cond_feat = aux_info['cond_feat']
        z= self.vae(x_start,cond_feat)
        noise_init = torch.randn_like(z)
        # noise_init = torch.randn_like(x)
        #noise = noise_init
        #x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #t_inp = t

        # if self.diffuser_input_mode == 'state_and_action':
        #     x_action_noisy = x_noisy[..., [4, 5]]
        #     x_noisy = self.convert_action_to_state_and_action(x_action_noisy, aux_info['curr_states'])







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


