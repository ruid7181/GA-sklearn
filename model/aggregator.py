from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from configurations.model_config import GADefaultHyperParameters as GAParams
from model.attention import CartesianPerceiver, VanillaPerceiver
from model.customs import NanBatchNorm1dNaive


class FCNTokenizer(nn.Module):
    """
    FCN encodes raw features to higher-dimensional (d_model // 2) features.
    Not encoding spa_feats, where we use 2-D rotary PE instead.
    """

    def __init__(self, d_model, x_dims, y_dims):
        self.d_model = d_model
        self.x_dims, self.y_dims = x_dims, y_dims
        super(FCNTokenizer, self).__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(len(self.x_dims), self.d_model // 2, bias=True),
            nn.Tanhshrink(),
            nn.Linear(self.d_model // 2, self.d_model // 2, bias=True)
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(len(self.y_dims), self.d_model // 2, bias=True),
            nn.Tanhshrink(),
            nn.Linear(self.d_model // 2, self.d_model // 2, bias=True)
        )
        self.learnable_y = nn.Parameter(torch.randn(1, 1, self.d_model // 2))

    def forward(self, x_tensor, y_tensor):
        """
        :param x_tensor: [bs, sl, x_dim]
        :param y_tensor: [bs, sl-1, 1]
        """
        batch_size, seq_len = x_tensor.shape[0], x_tensor.shape[1]
        x_embed = self.x_encoder(x_tensor)  # -> [bs, sl, dm//2]
        y_embed = self.y_encoder(y_tensor)  # -> [bs, sl-1, dm//2]

        y_embed = torch.concat((
            y_embed,
            self.learnable_y.repeat(batch_size, 1, 1)
        ), dim=1)  # -> [bs, sl, dm//2]

        return x_embed, y_embed  # [bs, sl, dm//2], [bs, sl, dm//2]


class FCNRegressor(nn.Module):
    def __init__(self, linear_dims):
        super(FCNRegressor, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(linear_dims[lyr], linear_dims[lyr + 1]) for lyr in range(len(linear_dims) - 1)
        ])

    def forward(self, encoding_tensor):
        """
        :param encoding_tensor: encoder output. [bs, 1, dm]
        :return: regression result [bs, 1, 1]
        """
        in_mat = encoding_tensor
        for lyr in self.decoder[:-1]:
            in_mat = lyr(in_mat)
            in_mat = F.tanhshrink(in_mat)

        in_mat = self.decoder[-1](in_mat)

        return in_mat


class RotaryEmbedding2D(nn.Module):
    """
    Optimized 2D Rotary Positional Embeddings for geospatial coordinates.
    Key optimizations:
    1. Pre-compute and cache frequency bands
    2. Use efficient batch operations
    3. Support both training and inference with minimal overhead
    4. Maintain compatibility with existing GeoAggregator architecture
    """

    def __init__(self, d: int = 4, base: int = 10000, scale: float = 1.0, n_heads: int = 4):
        """
        Args:
            d: Model dimension (must be divisible by 4 for 2D coordinates)
            base: Base for the frequency bands
            scale: Optional scaling factor for positions
            n_heads: Number of attention heads (default: 4)
        """
        super().__init__()
        assert d % 4 == 0, f"Model dimension {d} must be divisible by 4 for 2D coordinates"
        
        self.d = d
        self.base = base
        self.scale = scale
        self.n_heads = n_heads
        
        # Cache frequency bands - shape: [d//4] for each coordinate
        inv_freq = 1.0 / (base ** (torch.arange(0, d//4, 1).float() / (d//4)))
        self.register_buffer("inv_freq", inv_freq)
        
        # # Cache for inference
        # self.cached_shape = None
        # self.cached_sin_cos = None

    def _compute_sin_cos(self, coords: torch.Tensor) -> tuple:
        """
        Compute sin and cos values for given coordinates efficiently.
        Args:
            coords: [batch_size, seq_len, 2] coordinates tensor
        Returns:
            tuple of (sin_x, cos_x, sin_y, cos_y) each with shape [batch_size, seq_len, d//4]
        """
        x, y = coords[..., 0], coords[..., 1]
        batch_size, seq_len = x.shape
        
        # Scale and reshape coordinates
        x = x.view(batch_size, seq_len, 1) * self.scale
        y = y.view(batch_size, seq_len, 1) * self.scale
        
        # Compute frequency products efficiently
        freq_x = x @ self.inv_freq.view(1, -1)  # [batch_size, seq_len, d//4]
        freq_y = y @ self.inv_freq.view(1, -1)
        
        # Compute sin and cos
        sin_x, cos_x = torch.sin(freq_x), torch.cos(freq_x)
        sin_y, cos_y = torch.sin(freq_y), torch.cos(freq_y)
        
        return sin_x, cos_x, sin_y, cos_y

    def embed(self, spa_feat: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrices for spatial coordinates.
        Args:
            spa_feat: [batch_size, seq_len, 2] coordinates tensor
        Returns:
            Rotation matrix R [batch_size, seq_len, d, d]
        """
        batch_size, seq_len, _ = spa_feat.shape
        
        # # Check if we can use cached computations for inference
        # if not self.training:
        #     current_shape = (spa_feat.shape, spa_feat.device, spa_feat.dtype)
        #     if self.cached_shape == current_shape and self.cached_sin_cos is not None:
        #         sin_emb, cos_emb = self.cached_sin_cos
        #         return self._construct_rotation_matrix(sin_emb, cos_emb, batch_size, seq_len)
        
        # Compute sin and cos values
        sin_x, cos_x, sin_y, cos_y = self._compute_sin_cos(spa_feat)
        
        # Combine x and y components
        sin_emb = torch.cat([sin_x, sin_y], dim=-1)  # [batch_size, seq_len, d//2]
        cos_emb = torch.cat([cos_x, cos_y], dim=-1)
        
        # # Cache for inference
        # if not self.training:
        #     self.cached_shape = (spa_feat.shape, spa_feat.device, spa_feat.dtype)
        #     self.cached_sin_cos = (sin_emb, cos_emb)
        
        # Construct rotation matrix
        return self._construct_rotation_matrix(sin_emb, cos_emb, batch_size, seq_len)

    def _construct_rotation_matrix(self, sin_emb: torch.Tensor, cos_emb: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Construct rotation matrix from sin and cos embeddings.
        Args:
            sin_emb: [batch_size, seq_len, d//2]
            cos_emb: [batch_size, seq_len, d//2]
            batch_size: Batch size
            seq_len: Sequence length
        Returns:
            Rotation matrix [batch_size, seq_len, d, d]
        """
        # The sin and cos embeddings have shape [batch_size, seq_len, d//2]
        # For a proper 2x2 rotation matrix, we need to create a block diagonal matrix
        
        # First create block-wise matrices
        d_half = sin_emb.shape[-1]  # This is d//2
        
        # For each position (batch element, sequence position), we'll create 
        # a matrix of shape [d//2, d//2] where each 2x2 block is a rotation matrix
        
        # Initialize an empty rotation matrix
        R = torch.zeros(batch_size, seq_len, self.d, self.d, device=sin_emb.device)
        
        # Fill in the rotation matrices
        for i in range(d_half // 2):  # For each 2x2 block
            # Get the sin and cos values for this block
            sin_block = sin_emb[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            cos_block = cos_emb[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            
            # Compute 2x2 rotation matrix for this block
            block = torch.cat([
                torch.cat([cos_block, -sin_block], dim=-1),
                torch.cat([sin_block, cos_block], dim=-1)
            ], dim=-2)  # [b, s, 2, 2]
            
            # Place this block along the diagonal
            row_idx = i * 2
            col_idx = i * 2
            R[:, :, row_idx:row_idx+2, col_idx:col_idx+2] = block
        
        # Similarly for the y-coordinate blocks
        for i in range(d_half // 2):
            # Get the sin and cos values for this block
            sin_block = sin_emb[:, :, i + d_half//2].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            cos_block = cos_emb[:, :, i + d_half//2].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            
            # Compute 2x2 rotation matrix for this block
            block = torch.cat([
                torch.cat([cos_block, -sin_block], dim=-1),
                torch.cat([sin_block, cos_block], dim=-1)
            ], dim=-2)  # [b, s, 2, 2]
            
            # Place this block along the diagonal
            row_idx = (i + d_half//2) * 2
            col_idx = (i + d_half//2) * 2
            R[:, :, row_idx:row_idx+2, col_idx:col_idx+2] = block
            
        return R

    def __repr__(self):
        return f"RotaryEmbedding2D(d={self.d}, base={self.base}, scale={self.scale}, n_heads={self.n_heads})"


class GeoAggregator(nn.Module):
    def __init__(self,
                 x_dims: tuple,
                 spa_dims: tuple,
                 y_dims: tuple,
                 attn_variant: Literal['MCPA', 'vanilla'] = 'MCPA',
                 model_variant: Optional[Literal['mini', 'small', 'large']] = None,
                 d_model: int = 32,
                 n_attn_layer: int = 2,
                 idu_points: int = 4,
                 attn_dropout: float = 0.2,
                 attn_bias_factor: float = None,
                 dc_lin_dims: list = None):
        """
        An encoder-processor-decoder architecture (see GeoAggregator paper).

        :param x_dims:
            Dim indices for x features (co-variates).
        :param spa_dims:
            Dim indices for spatial coordinates.
        :param y_dims:
            Dim indices for y feature dims.
        :param attn_variant:
            Variant of the attention mechanism: Cartesian attention (MCPA) or
            vanilla attention (see GeoAggregator paper).
        :param model_variant:
            Variant of the GeoAggregator model (see GeoAggregator paper).
            When this param is not None, params `n_attn_layer` and `idu_points`
            are ignored.
        :param d_model:
            Embedding dimension throughout the transformer model.
            Note that, as indicated in the paper (section 3.4), the following equation
            holds:
                2 * d_c * H^2 = d_{model}
            In our implementation, we set H = 2, thus the hyperparameter `d_model`
            must be divisible by 8.
        :param n_attn_layer:
            Number of total attention layers. Default = 2.
            - When n_attn_layer = 1, GA aggregates the input points into 1
            point, without further decoding.
            - When n_attn_layer = 2, the first attention layer (the encoder)
            aggregates the input points into `idu_points`-many inducing points;
            the second attention layer (the decoder) aggregates the
            inducing points into one point, before final regression.
            - When n_attn_layer = n (n >= 3), the first attention layer (the encoder)
            aggregates the input points into `idu_points`-many inducing points;
            the following n-2 attention layers (the processor) learn the interaction
            between these inducing points; the final attention layer (the decoder)
            aggregates the inducing points into one point, before final regression.
        :param idu_points:
            Number of inducing points.
        :param attn_dropout:
            Dropout probability.
        :param attn_bias_factor:
            If the attention bias factor is not given (None), the model dynamically
            learns the optimized value for the attention bias factor.
        :param dc_lin_dims:
            Decoder ffn dims.
        """
        super(GeoAggregator, self).__init__()
        # ----------------------------------------------------------------
        self.x_dims = x_dims
        self.spa_dims = spa_dims
        self.y_dims = y_dims
        self.attn_variant = attn_variant

        if model_variant is not None:
            idu_points = GAParams.variants[model_variant]['idu_points']
            n_attn_layer = GAParams.variants[model_variant]['n_attn_layer']

        # Check dimensions:
        assert d_model % 8 == 0, ('`d_model` must be divisible by 8. '
                                  'Refer to the docstring for details.')
        # ----------------------------------------------------------------
        # Encoder-Processor-Decoder architecture
        self._n_batch_norm = NanBatchNorm1dNaive(
            n_feature=len(self.x_dims + self.y_dims)
        )
        self._tokenizer = FCNTokenizer(
            d_model=d_model,
            x_dims=self.x_dims,
            y_dims=self.y_dims
        )
        self._rotary_embed = RotaryEmbedding2D(
            d=d_model // 8,
            base=10000,
            scale=1.0,
            n_heads=4
        )
        self._perceiver = CartesianPerceiver(
            d_model=d_model,
            n_attn_layer=n_attn_layer,
            attn_dropout=attn_dropout,
            attn_bias_factor=attn_bias_factor,
            n_hidden_token=idu_points
        )
        if attn_variant == 'MCPA':
            self._perceiver = CartesianPerceiver(
                d_model=d_model,
                n_attn_layer=n_attn_layer,
                attn_dropout=attn_dropout,
                attn_bias_factor=attn_bias_factor,
                n_hidden_token=idu_points
            )
        else:
            self._perceiver = VanillaPerceiver(
                d_model=d_model,
                n_head=4,
                n_attn_layer=n_attn_layer,
                n_hidden_token=idu_points,
                attn_dropout=attn_dropout,
                attn_bias_factor=attn_bias_factor
            )
        # ----------------------------------------------------------------
        # Regressor
        self._dc_lin_dims = dc_lin_dims.copy()
        self._dc_lin_dims.insert(
            0, d_model + len(self.x_dims + self.spa_dims)
        )
        self._regressor = FCNRegressor(linear_dims=self._dc_lin_dims)

    def __enc_proc_dec(self,
                       input_tensor,
                       mask,
                       geo_proximity):
        """
        The Encoder-Processor-Decoder.
        :param input_tensor: [bs, sl, fd].
        :param mask: [bs, sl-1], masked: 1, not masked: 0.
        :param geo_proximity: [bs, sl-1], distances.
        """
        # 2D Rotary Positional Embedding
        # -> [bs, sl, 2]
        spa_embed = input_tensor[:, :, self.spa_dims]
        # [bs, sl, 2] -> [bs, sl, dk, dk]
        spa_embed = self._rotary_embed.embed(spa_embed)

        # Feature Projection
        # -> [bs, sl, x_dims]
        x_embed = input_tensor[:, :, self.x_dims]
        # -> [bs, sl-1, y_dims]
        y_embed = input_tensor[:, :-1, self.y_dims]
        # -> [bs, sl, dm//2], [bs, sl, dm//2]
        x_embed, y_embed = self._tokenizer(x_tensor=x_embed, y_tensor=y_embed)

        # Attention
        if self.attn_variant == 'MCPA':
            # -> [2, bs, ql(1), dm//2]
            return self._perceiver(
                a_embed=x_embed[:, :-1, :],
                b_embed=y_embed[:, :-1, :],
                q_a_embed=x_embed[:, -1:, :],
                q_b_embed=y_embed[:, -1:, :],
                ctx_spa_embed=spa_embed[:, :-1, :, :],
                q_spa_embed=spa_embed[:, -1:, :, :],
                mask=mask,
                geo_proxy=geo_proximity
            )
        else:
            # -> [bs, sl-1, dm]
            ctx_embed = torch.cat(
                (x_embed[:, :-1, :], y_embed[:, :-1, :]),
                dim=-1
            )
            q_embed = torch.cat(
                (x_embed[:, -1:, :], y_embed[:, -1:, :]),
                dim=-1
            )

            # -> [bs, ql(1), dm]
            return self._perceiver(
                ctx_embed=ctx_embed,
                q_embed=q_embed,
                ctx_spa_embed=spa_embed[:, :-1, :, :],
                q_spa_embed=spa_embed[:, -1:, :, :],
                mask=mask,
                geo_proxy=geo_proximity
            )

    def forward(self,
                input_tensor,
                geo_proximity,
                get_attn_score=False):
        """
        :param geo_proximity: [bs, sl]
        :param input_tensor: [bs, sl, x_dims + y_dims + spa_dims]
        :param get_attn_score: return attention scores, True or False.
        """
        # Preprocess
        mask = torch.where(input_tensor.isnan(), 1, 0)[:, :-1, 0]  # -> [bs, sl-1]
        input_tensor[:, :, self.x_dims + self.y_dims] = \
            self._n_batch_norm(input_tensor[:, :, self.x_dims + self.y_dims])

        input_tensor = input_tensor.nan_to_num()
        geo_proximity = geo_proximity.nan_to_num()[:, :-1]  # -> [bs, sl] -> [bs, sl-1]

        target_point_x = input_tensor[:, -1][:, :-1].clone()  # -> [bs, fd] -> [bs, fd-1]
        target_point_x = target_point_x.unsqueeze(1)  # [bs, fd-1] -> [bs, 1, fd-1]

        # Perceiver
        encoding_tensor, attn_weights = self.__enc_proc_dec(
            input_tensor=input_tensor,
            mask=mask,
            geo_proximity=geo_proximity
        )

        # Regression
        _, batch_size, query_len, d_model_half = encoding_tensor.shape
        encoding_tensor = encoding_tensor.permute(1, 2, 0, 3).contiguous().view(batch_size, query_len, -1)
        encoding_tensor = torch.concat((encoding_tensor, target_point_x), dim=-1)

        pred = self._regressor(encoding_tensor=encoding_tensor)  # -> [bs, 1, 1]

        if get_attn_score:
            return pred, attn_weights
        else:
            return pred
