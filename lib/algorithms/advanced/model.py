import functools

import torch
import torch.nn as nn

from lib.utils.generic import import_configs
from lib.algorithms.advanced.module import GaussianFourierProjection, get_sigmas, get_timestep_embedding, get_act, \
    TimestepEmbedder, PositionalEncoding


def create_model(model_config, N_POSES, POSE_DIM):
    if 'FC' in model_config.type:
        Model = MaskFC if 'Mask' in model_config.type else TimeFC
        model = Model(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            embed_dim=model_config.EMBED_DIM,
            n_blocks=model_config.N_BLOCKS,
        )
    elif model_config.type == 'TimeMLPs':
        model = TimeMLPs(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            n_blocks=model_config.N_BLOCKS,
        )
    elif 'Transformer' in model_config.type:
        model = MaskTransformer(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
        )
    else:
        raise NotImplementedError('unsupported model')

    return model


class TimeMLPs(torch.nn.Module):
    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64, n_blocks=2):
        super().__init__()
        dim = n_poses * pose_dim
        self.act = get_act(config)

        layers = [torch.nn.Linear(dim + 1, hidden_dim),
                  self.act]

        for _ in range(n_blocks):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                self.act,
                torch.nn.Dropout(p=config.dropout)
            ])

        layers.append(torch.nn.Linear(hidden_dim, dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, batch, t, condition=None, mask=None):
        return self.net(torch.cat([batch, t[:, None]], dim=1))


class TimeFC(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, model_config, n_poses=21, pose_dim=6, hidden_dim=64,
                 embed_dim=32, n_blocks=2):
        super(TimeFC, self).__init__()
        self.model_config = model_config
        self.n_poses = n_poses
        self.joint_dim = pose_dim
        self.n_blocks = n_blocks

        self.act = get_act(model_config)

        self.pre_dense = nn.Linear(n_poses * pose_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=model_config.dropout)

        # time embedding
        self.time_embedding_type = model_config.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim, scale=model_config.fourier_scale)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas(model_config), dtype=torch.float))

        for idx in range(n_blocks):
            setattr(self, f'b{idx + 1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx + 1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

        self.post_dense = nn.Linear(hidden_dim, n_poses * pose_dim)

    def forward(self, batch, t, condition=None, mask=None):
        """
        batch: [B, j*3] or [B, j*6]
        t: [B]
        condition: not be enabled
        mask: [B, j*3] or [B, j*6] same dim as batch
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        bs = batch.shape[0]

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)

        h = self.pre_dense(batch)
        h += self.pre_dense_t(temb)
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)

        for idx in range(self.n_blocks):
            h1 = getattr(self, f'b{idx + 1}_dense1')(h)
            h1 += getattr(self, f'b{idx + 1}_dense1_t')(temb)
            h1 = getattr(self, f'b{idx + 1}_gnorm1')(h1)
            h1 = self.act(h1)
            # dropout, maybe
            h1 = self.dropout(h1)

            h2 = getattr(self, f'b{idx + 1}_dense2')(h1)
            h2 += getattr(self, f'b{idx + 1}_dense2_t')(temb)
            h2 = getattr(self, f'b{idx + 1}_gnorm2')(h2)
            h2 = self.act(h2)
            # dropout, maybe
            h2 = self.dropout(h2)

            h = h + h2

        res = self.post_dense(h)  # [B, j*3]

        ''' normalize the output '''
        if self.model_config.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1))
            res = res / used_sigmas

        return res


class MaskFC(TimeFC):
    def __init__(self, model_config, n_poses=21, pose_dim=6, hidden_dim=64,
                 embed_dim=32, n_blocks=2, vec_dim=10):  # 添加了vec_dim参数
        super(MaskFC, self).__init__(model_config, n_poses, pose_dim, hidden_dim,
                                     embed_dim, n_blocks)
        self.model_config = model_config
        self.masked_vector = nn.Parameter(torch.randn(vec_dim))
        self.unmasked_vector = nn.Parameter(torch.randn(vec_dim))

        self.pre_dense = nn.Linear(n_poses * pose_dim * (1 + vec_dim), hidden_dim)

    def forward(self, batch, t, condition=None, mask=None, ):
        if mask is None:
            mask = batch.new_ones(batch.shape)
        bs = batch.shape[0]
        # Mask batching
        batch = batch * mask
        joint_vector = torch.where(mask[..., None] == 1, self.unmasked_vector, self.masked_vector)
        masked_batch = torch.cat([batch[..., None], joint_vector], dim=-1).reshape(bs, -1)  # [b, j*3*(1+vec_dim)]

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)

        h = self.pre_dense(masked_batch)
        h += self.pre_dense_t(temb)
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)

        for idx in range(self.n_blocks):
            h1 = getattr(self, f'b{idx + 1}_dense1')(h)
            h1 += getattr(self, f'b{idx + 1}_dense1_t')(temb)
            h1 = getattr(self, f'b{idx + 1}_gnorm1')(h1)
            h1 = self.act(h1)
            # dropout, maybe
            h1 = self.dropout(h1)

            h2 = getattr(self, f'b{idx + 1}_dense2')(h1)
            h2 += getattr(self, f'b{idx + 1}_dense2_t')(temb)
            h2 = getattr(self, f'b{idx + 1}_gnorm2')(h2)
            h2 = self.act(h2)
            # dropout, maybe
            h2 = self.dropout(h2)

            h = h + h2

        res = self.post_dense(h)  # [B, j*3]

        ''' normalize the output '''
        if self.model_config.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1))
            res = res / used_sigmas

        return res


class MaskTransformer(nn.Module):
    def __init__(self, model_config, n_poses=21, pose_dim=3):
        super().__init__()
        self.model_config = model_config
        self.n_poses = n_poses
        self.pose_dim = pose_dim
        self.hidden_dim = model_config.HIDDEN_DIM
        self.ff_dim = model_config.FF_DIM
        self.n_heads = model_config.N_HEADS
        self.n_layers = model_config.N_LAYERS
        self.dropout = model_config.dropout

        # Linear layers to transform input to hidden_dim
        self.input_processor = nn.Linear(self.pose_dim, self.hidden_dim)
        self.position_encoder = PositionalEncoding(self.hidden_dim, dropout=self.dropout)

        # time embedding
        self.time_embedding_type = model_config.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.time_pe = GaussianFourierProjection(embed_dim=self.hidden_dim, scale=model_config.fourier_scale)
        elif self.time_embedding_type == 'positional':
            self.time_pe = functools.partial(get_timestep_embedding, embedding_dim=self.hidden_dim)
        else:
            assert 0
        self.time_processor = TimestepEmbedder(self.hidden_dim, self.time_pe)
        self.register_buffer('sigmas', torch.tensor(get_sigmas(model_config), dtype=torch.float))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.n_heads,
                                                   dim_feedforward=self.ff_dim, dropout=self.dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Linear layer to transform back to pose_dim
        self.output_linear = nn.Linear(self.hidden_dim, self.pose_dim)

    def forward(self, batch, t, condition=None, mask=None):
        B, _ = batch.shape

        # Transform to hidden_dim
        batch = batch.view(B, self.n_poses, self.pose_dim)  # [B, j, 3]
        batch = self.position_encoder(self.input_processor(batch))  # [B, j, hidden_dim]
        temb = self.time_processor(t).unsqueeze(1)  # [B, hidden_dim] -> [B, 1, hidden_dim]

        # Concatenate batch and temb
        seq = torch.cat([batch, temb], dim=1)  # [B, j+1, hidden_dim]

        # Process mask
        if mask is not None:
            mask = mask[:, ::3]  # [B, j]
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=mask.device)], dim=1)  # [B, j+1]
            mask = torch.logical_not(mask)  # Invert mask

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(seq, src_key_padding_mask=mask)  # [B, j+1, hidden_dim]

        # Extract batch output and transform back to pose_dim
        batch_output = transformer_output[:, :-1, :]  # [B, j, hidden_dim]
        batch_output = self.output_linear(batch_output).view(B, -1)  # [B, j, 3] -> [B, j*3]

        ''' normalize the output '''
        if self.model_config.scale_by_sigma:
            if self.time_embedding_type == 'fourier':
                used_sigmas = t
            elif self.time_embedding_type == 'positional':
                used_sigmas = self.sigmas[t.long()]
            else:
                raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')
            used_sigmas = used_sigmas.reshape((B, 1))
            batch_output = batch_output / used_sigmas

        return batch_output


if __name__ == '__main__':
    from lib.dataset.body import N_POSES
    config_path = 'configs.subvp.amass_scorefc_continuous.get_config'
    config = import_configs(config_path)
    # model = ScoreModelFC(config,
    #                      n_poses=N_POSES,
    #                      pose_dim=3,
    #                      hidden_dim=config.model.HIDDEN_DIM,
    #                      embed_dim=config.model.EMBED_DIM,
    #                      n_blocks=config.model.N_BLOCKS,
    #                      )
    model = MaskTransformer(config,
                            n_poses=N_POSES,
                            pose_dim=3,
                            )

    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis

    batchsize = 1
    summary(model, [(batchsize, 63), (batchsize,)], dtypes=[torch.float, torch.float], mode="train")

    inputs = {'batch': torch.randn(batchsize, 63, device='cuda:0'),
              't': torch.ones([batchsize], device='cuda:0'), }
    outputs = model(**inputs)
    print(outputs.shape)

    # 使用fvcore计算FLOPs
    flops = FlopCountAnalysis(model, (inputs['batch'], inputs['t']))
    print(f"FLOPs: {flops.total()}")

    '''
    TimeMLPs
    ==========================================================================================
    Total params: 2,230,335
    Trainable params: 2,230,335
    Non-trainable params: 0
    Total mult-adds (M): 2.23
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.03
    Params size (MB): 8.92
    Estimated Total Size (MB): 8.95
    ==========================================================================================

    ScoreModelFC
    ==========================================================================================
    Total params: 7,227,967
    Trainable params: 7,227,967
    Non-trainable params: 0
    Total mult-adds (M): 7.23
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.13
    Params size (MB): 28.91
    Estimated Total Size (MB): 29.04
    ==========================================================================================

    MaskScoreModelFC
    ==========================================================================================
    Total params: 7,873,107
    Trainable params: 7,873,107
    Non-trainable params: 0
    Total mult-adds (M): 7.87
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.13
    Params size (MB): 31.49
    Estimated Total Size (MB): 31.62
    ==========================================================================================
    
    MaskTransformer
    ===============================================================================================
    Total params: 3,292,419
    Trainable params: 3,292,419
    Non-trainable params: 0
    Total mult-adds (M): 2.24 (70.6)
    ===============================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.31
    Params size (MB): 8.96
    Estimated Total Size (MB): 10.27
    ===============================================================================================
    '''
