import math

import torch
import torch.nn as nn

from ts_sum.s3dg import S3D


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class VSum_MLP(nn.Module):
    def __init__(
            self,
            num_classes=512,
            gating=True,
            space_to_depth=False,
            word2vec_path="",
            init="uniform",
            token_to_word_path="data/dict.npy",
            d_model=512,
            dropout=0.1,
    ) -> None:
        super(VSum_MLP, self).__init__()
        self.base_model = S3D(
            num_classes, space_to_depth=True, word2vec_path=word2vec_path, init=init,
        )
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, video):
        # [B, C, S, H, W] -> [B, S, C, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        H, W = video.shape[3], video.shape[4]

        # [B, S, C, H, W] -> [B*S, C, H, W]
        video = video.contiguous().view(-1, 3, H, W)
        n_segs = int(video.shape[0] / 32)

        # [B*S, C, H, W] -> [n_segs, 32, C, H, W]
        video = video.view(n_segs, 32, 3, H, W)

        # [n_segs, T, C, H, W] -> [n_segs, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)

        # [n_segs, T, C, H, W] -> [n_segs, C, T, H, W]
        video_emb = self.base_model.forward_video(video)

        x = self.mlp(video_emb)
        x = x.contiguous().view(-1, 256)
        logits = self.fc(x)

        return video_emb, logits


class VSum(nn.Module):
    def __init__(
            self,
            num_classes=512,
            gating=True,
            space_to_depth=False,
            word2vec_path="",
            init="uniform",
            token_to_word_path="data/dict.npy",
            window_len=32,
            heads=8,
            enc_layers=6,
            d_model=512,
            dropout=0.1,
    ) -> None:
        super(VSum, self).__init__()
        self.window_len = window_len

        self.base_model = S3D(
            num_classes, space_to_depth=space_to_depth, word2vec_path=word2vec_path, init=init,
        )

        self.d_model = d_model
        self.pos_enc = PositionalEncoding(self.d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=enc_layers
        )
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, video):
        # [B, C, S, H, W] -> [B, S, C, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        H, W = video.shape[3], video.shape[4]

        # [B, S, C, H, W] -> [B*S, C, H, W]
        video = video.contiguous().view(-1, 3, H, W)
        n_segs = int(video.shape[0] / self.window_len)

        # [B*S, C, H, W] -> [n_segs, 32, C, H, W]
        video = video.view(n_segs, self.window_len, 3, H, W)

        # [n_segs, T, C, H, W] -> [n_segs, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)

        # [n_segs, T, C, H, W] -> [n_segs, C, T, H, W]
        video_emb = self.base_model.forward_video(video)

        # [n_segs, d_model] -> [1, n_segs, d_model]
        video_emb = video_emb.unsqueeze(0)

        video_emb = self.pos_enc(
            video_emb
        )  # Add pos enc as nn.Trasnformer doesnt have it

        video_emb = self.transformer_encoder(video_emb)
        video_emb = video_emb.contiguous().view(-1, self.d_model)
        logits = self.fc(video_emb)

        return video_emb, logits
