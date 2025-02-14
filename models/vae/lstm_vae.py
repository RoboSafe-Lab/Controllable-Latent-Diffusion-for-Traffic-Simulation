
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,cond_dim=256,dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.cond2hidden = nn.Linear(cond_dim, hidden_size)
    def forward(self, x,context):
        batch_size = x.size(0)
        cond_hidden = self.cond2hidden(context) #[B,64]
        h0 = cond_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [2, B, hidden_size]
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        outputs, (hn, cn) = self.lstm(x, (h0, c0))
        return outputs

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.2,cond_dim=256):
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
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.cond2hidden = nn.Linear(cond_dim, hidden_size)
        self.hid2act = nn.Linear(hidden_size,output_size)
    def forward(self, x, context):
        batch_size = x.size(0)
        cond_hidden = self.cond2hidden(context)
        h0 = cond_hidden.unsqueeze(0).repeat(self.num_layers,1,1)#[2,B,64]
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        decode_output, (hn, cn) = self.lstm(x, (h0, c0))#[B,52,64]
        output = self.hid2act(decode_output)

        return output

class LSTMVAE(nn.Module):
    def __init__(
        self, input_size, hidden_size, latent_size, output_size,dropout_rate=0.2,device=torch.device("cuda")
    ):
        super(LSTMVAE, self).__init__()
        self.device = device
        self.input_size = input_size #6
        self.hidden_size = hidden_size #64
        self.latent_size = latent_size #4
        self.num_layers = 2

        self.lstm_enc = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,#64
            num_layers=self.num_layers,
            dropout_rate=dropout_rate
        )
        self.lstm_dec = Decoder(
            input_size=latent_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=self.num_layers,
            dropout_rate=dropout_rate,
        )

        self.mu = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.hidden_size, self.latent_size)
   
    def forward(self,x,context):
        z,mean,logvar = self.traj2z(x,context) #[B,T,4]
        act_output = self.lstm_dec(z,context)
        return act_output,mean,logvar

    def traj2z(self, x, context):
        B, T, _ = x.shape
        env_outputs = self.lstm_enc(x,context) #  [B, T, hidden_size=64]
        mean = self.mu(env_outputs) #[B,T,4]
        logvar = self.logvar(env_outputs)
        z = self.reparametize(mean,logvar)#[B,T,4]
        return z,mean,logvar

    def reparametize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)
        z = mean + noise * std
        return z

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

        kld_loss = torch.mean( -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1) )

   
        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
        }

    