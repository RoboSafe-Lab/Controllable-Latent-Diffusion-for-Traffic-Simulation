from tbsim.models.diffuser import DiffuserModel
from  models.vae import VAE

class DMVAE(DiffuserModel):
    def __init__(
        self,
        vae_hidden_dims: tuple = (128, 256, 512),
        vae_loss_weight: float = 1.0,
        **kwargs
    ):
        super(DMVAE, self).__init__(**kwargs)
        self.vae = VAE()



        self.vae_loss_weight = vae_loss_weight

    def forward(self, x, aux_info, time):

        dm_output = super().forward(x, aux_info, time)

        x_flattened = dm_output.view(dm_output.size(0), -1)
        latent = self.vae_encoder(x_flattened)
        recon = self.vae_decoder(latent)

        recon = recon.view_as(dm_output)

        return dm_output, recon, latent

    def compute_dmvae_loss(self, recon, target, latent):

       pass


