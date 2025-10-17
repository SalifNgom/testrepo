import torch
import torch.nn as nn

class CausalSliceTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, num_layers=4, num_heads=8, num_slices=9,alpha=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_slices = num_slices
        self.alpha=alpha
        # Encodage slice → embedding vector
        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(2),  # [B, D, 1]
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)  # Optionnel
        )


        # Embedding de position appris
        self.pos_embed = nn.Parameter(torch.randn(1, num_slices, embed_dim))

        # Transformer causal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reprojection avec ConvTranspose2d
        self.decoder_proj = nn.Sequential(
            nn.Linear(embed_dim, 256 * 6 * 6),         # [B, 9216]
            nn.Unflatten(1, (256, 6, 6)),              # [B, 256, 6, 6]
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B, 128, 12, 12]
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # [B, 64, 24, 24]
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1),  # [B, 3, 48, 48]
        )


    def forward(self, patches):  
        """
        patches: [B, T, C, H, W] — slices encodées
        returns: [B, T, C, H', W'] — slices enrichies avec continuité causale
        """
        B, T, C, H, W = patches.shape
        x = patches.view(B*T, C, H, W)                  # [B*T, C, H, W]
        x = self.patch_proj(x).squeeze(-1)              # [B*T, D]
        x = x.view(B, T, -1)                            # [B, T, D]

        x = x + self.pos_embed                          # Position
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        out = self.transformer(x, mask=causal_mask)     # [B, T, D]

        out_flat = out.view(B*T, -1)                    # [B*T, D]
        out = self.decoder_proj(out_flat)       # [B*T, C, H', W']
        _, C, H, W = out.shape
        return  self.alpha*out.view(B, T, C, H, W) +patches









#for batch in dataloader:
 #   x = batch.to(device)
  #  output = model(x)
    # Ignorer la slice t=0 pour le loss
   # loss = sum([criterion(output[:, t], x[:, t]) for t in range(1, T)]) / (T - 1)

    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
