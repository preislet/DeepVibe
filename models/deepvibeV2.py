import torch
import torch.nn as nn

class HybridDeepVibeVAE(nn.Module):
    def __init__(self, interaction_dim=10000, audio_dim=12, latent_dim=256):
        super(HybridDeepVibeVAE, self).__init__()
        
        # Encoder: Bere interakce + průměrný audio profil uživatele
        combined_input_dim = interaction_dim + audio_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(combined_input_dim, 1024),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: Rekonstruuje interakce (předpovídá ideální poslech)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SELU(),
            nn.Linear(256, 512),
            nn.SELU(),
            nn.Linear(512, 1024),
            nn.SELU(),
            nn.Linear(1024, interaction_dim) 
        )
        
        # Inicializace vah pro SELU (LeCun Normal)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x_int, x_audio):
        # Spojení interakcí a audio profilu (2D vstup)
        combined = torch.cat([x_int, x_audio], dim=1)
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        # Bezpečnostní prvek proti výbuchu gradientu
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_int, x_audio):
        mu, logvar = self.encode(x_int, x_audio)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar