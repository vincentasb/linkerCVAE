import torch
import torch.nn.functional as F
import torch.nn as nn

class CVAE(torch.nn.Module):
    def __init__(self, input_size=420):
        super().__init__()

        encoder_layer_sizes = [390, 256, 128, 16]
        decoder_layer_sizes = [46, 128, 256, 390]

        # Encoder, leaky relu, mabye embbeding layeri prikliuot, 
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_layer_sizes[0]),
            nn.BatchNorm1d(encoder_layer_sizes[0]),
            nn.LeakyReLU(0.01),
            nn.Linear(encoder_layer_sizes[0], encoder_layer_sizes[1]),
            nn.BatchNorm1d(encoder_layer_sizes[1]),
            nn.LeakyReLU(0.01),
            nn.Linear(encoder_layer_sizes[1], encoder_layer_sizes[2]),
            nn.BatchNorm1d(encoder_layer_sizes[2]),
            nn.LeakyReLU(0.01),
            nn.Linear(encoder_layer_sizes[2], encoder_layer_sizes[3])
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(decoder_layer_sizes[0], decoder_layer_sizes[1]),
            nn.BatchNorm1d(decoder_layer_sizes[1]),
            nn.LeakyReLU(0.01),
            nn.Linear(decoder_layer_sizes[1], decoder_layer_sizes[2]),
            nn.BatchNorm1d(decoder_layer_sizes[2]),
            nn.LeakyReLU(0.01),
            nn.Linear(decoder_layer_sizes[2], decoder_layer_sizes[3]),
            nn.BatchNorm1d(decoder_layer_sizes[3]),
            nn.LeakyReLU(0.01),
            nn.Linear(decoder_layer_sizes[3], 390),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, log_var):
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        #print(x.shape, c.shape)
        encoder_out = self.encoder(x)
        mu = encoder_out
        log_var = F.softplus(encoder_out)
        sample = self.reparametrize(mu, log_var)
        #print(sample.shape)
        sample = torch.cat((sample, c), 1)
        #print(sample.shape)
        decoder_out = self.decoder(sample)
        return decoder_out, mu, log_var
