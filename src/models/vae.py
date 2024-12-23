
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,cond_dim=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.cond2hidden = nn.Linear(cond_dim, hidden_size)
    def forward(self, x,cond_feature):
        # x: tensor of shape (batch_size, seq_length, feature_dim)
        batch_size = x.size(0)
        cond_hidden = self.cond2hidden(cond_feature)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        h0[0, :, :] = cond_hidden
        outputs, (hn, cn) = self.lstm(x, (h0, c0))
        return outputs

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, latent_size)
        output, _ = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction

class LSTMVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
        self, input_size, hidden_size, latent_size, output_size,device=torch.device("cuda")
    ):
        super(LSTMVAE, self).__init__()
        self.device = device

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = 1
      

        # lstm ae
        self.lstm_enc = Encoder(
            input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers,
        )
        self.lstm_dec = Decoder(
            input_size=latent_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=self.num_layers,
        )

        # Variational components
        self.mu = nn.Conv1d(self.hidden_size, self.latent_size, kernel_size=3, padding=1)
        self.var = nn.Conv1d(self.hidden_size, self.latent_size, kernel_size=3, padding=1)
        self.decoder_hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.decoder_cell = nn.Linear(self.latent_size, self.hidden_size)
      
        # self.scale = scale
        # self.descale = descale
    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)
        z = mu + noise * std
        return z

    def forward(self, x,aux_info):
        batch_size, _, _ = x.shape
        cond_feature = aux_info['cond_feat']
        
        # encode input space to hidden space
        enc_outputs = self.lstm_enc(x,cond_feature)

        # Prepare for latent space
        enc_outputs = enc_outputs.permute(0, 2, 1)  # [B, hidden_size, seq_len]
        mean = self.mu(enc_outputs)  # [B, latent_size, seq_len]
        logvar = self.var(enc_outputs)  # [B, latent_size, seq_len]

        # Reparametrize to get z
        z = self.reparametize(mean, logvar)  # [B, latent_size, seq_len]
        z = z.permute(0, 2, 1)  # [B, seq_len, latent_size]

        # Initialize decoder hidden state from z
        z_aggregated = z.mean(dim=1)  # Aggregate over time: [B, latent_size]
        h_0 = self.decoder_hidden(z_aggregated).view(self.num_layers, batch_size, self.hidden_size)#(1,B,128-hidden)
        c_0 = self.decoder_cell(z_aggregated).view(self.num_layers, batch_size, self.hidden_size)
        hidden = (h_0, c_0)

        # Decode latent space to input space
        reconstruct_output = self.lstm_dec(z, hidden)

        return reconstruct_output,mean,logvar

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = args[4]
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
    -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=(1, 2))
)

        #TODO:训练时,kl loss不下降,想办法
        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }
