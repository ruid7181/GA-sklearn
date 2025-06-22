import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_attention_bias(geo_proxy, attn_bias_factor):
    """
    :param geo_proxy: [bs, sl]
    :param attn_bias_factor: [4] (n-head = nh = 4)

    :return: [bs, nh, sl]
    """
    batch_size, seq_len = geo_proxy.shape
    geo_proxy **= 2
    max_, _ = geo_proxy.max(dim=1, keepdim=True)
    min_, _ = geo_proxy.min(dim=1, keepdim=True)
    geo_proxy = (geo_proxy - min_) / (max_ - min_ + 1e-8)
    if isinstance(attn_bias_factor, torch.Tensor):
        geo_proxy = torch.mul(
            geo_proxy.unsqueeze(1).repeat(1, 4, 1),   # [bs, sl] -> [bs, nh (4), sl]
            attn_bias_factor.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len)   # -> [bs, nh, sl]
        ) # -> [bs, nh, sl]
    else:
        geo_proxy *= attn_bias_factor

    return geo_proxy


class CartesianPerceiver(nn.Module):
    def __init__(self,
                 d_model,
                 n_attn_layer,
                 n_hidden_token=4,
                 attn_dropout=0.1,
                 attn_bias_factor=None):
        """
        A Perceiver-like architecture (Encoder-Processor-Decoder)
        Jaegle, A., et al. (2021). Perceiver io: A general architecture for structured inputs & outputs.

        :param d_model:
            Embedding dimension throughout the transformer model.
        :param n_attn_layer:
            Number of total attention layers.
        :param n_hidden_token:
            Number of hidden token, or inducing points.
        :param attn_dropout:
            Dropout probability for the attention mechanism.
        :param attn_bias_factor:
            Scaler controlling the amount of attention bias added to the attention
            (see GeoAggregator paper).
        """
        super(CartesianPerceiver, self).__init__()
        # ----------------------------------------------------------------
        self.d_model = d_model
        self.n_attn_layer = n_attn_layer
        self.n_hidden_token = n_hidden_token
        self.attn_dropout = attn_dropout
        # ----------------------------------------------------------------
        if attn_bias_factor:
            self.attn_bias_factor = attn_bias_factor
        else:
            self.attn_bias_factor = nn.Parameter(torch.zeros(4))

        self.attn_layers = nn.ModuleList([
            MaskedCartesianAttention(
                d_model=d_model,
                n_a_head=2,
                n_b_head=2,
                attn_dropout=self.attn_dropout,
                attn_bias_factor=self.attn_bias_factor
            )
            for _ in range(self.n_attn_layer)
        ])
        self.a_ffds = nn.ModuleList([
            nn.Linear(self.d_model // 2, self.d_model // 2)
            for _ in range(self.n_attn_layer)
        ])
        self.b_ffds = nn.ModuleList([
            nn.Linear(self.d_model // 2, self.d_model // 2)
            for _ in range(self.n_attn_layer)
        ])

        if self.n_attn_layer > 1:
            self.latent_queries_a = {
                n: nn.Parameter(torch.randn(self.n_hidden_token, self.d_model // 2))
                for n in range(self.n_attn_layer - 1)
            }
            self.latent_queries_b = {
                n: nn.Parameter(torch.randn(self.n_hidden_token, self.d_model // 2))
                for n in range(self.n_attn_layer - 1)
            }

        self.softmax = nn.Softmax(dim=3)

        self.a_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])  # layer_norm the last dimension
        self.b_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])

        self.a_ffd_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])  # layer_norm the last dimension
        self.b_ffd_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])

    def forward(self,
                a_embed,
                b_embed,
                q_a_embed,
                q_b_embed,
                ctx_spa_embed,
                q_spa_embed,
                mask,
                geo_proxy):
        """
        :param ctx_spa_embed: [bs, sl-1, dm//8, dm//8]
        :param q_spa_embed: [bs, 1, dm//8, dm//8]
        :param a_embed: [bs, sl-1, dm//2]
        :param b_embed: [bs, sl-1, dm//2]
        :param q_a_embed: [bs, 1, dm//2]
        :param q_b_embed: [bs, 1, dm//2]
        :param mask: [bs, sl-1], masked: 1, not masked: 0.
        :param geo_proxy: [bs, sl-1]
        """
        batch_size = ctx_spa_embed.shape[0]
        attn_weights = torch.FloatTensor([])
        gaussian_bias = _calc_attention_bias(geo_proxy, self.attn_bias_factor)  # -> [bs, nh, sl]

        embedding = torch.stack((a_embed, b_embed), dim=0)  # -> [2, bs, sl, dm//2]
        query = torch.stack((
            q_a_embed,
            q_b_embed
        ), dim=0)  # -> [2, bs, 1, dm//2]

        # Perceiver
        for idx, (atn,
                  a_ffd, b_ffd,
                  a_atn_ln, b_atn_ln,
                  a_ffd_ln, b_ffd_ln) in enumerate(zip(self.attn_layers[:-1],
                                                       self.a_ffds[:-1],
                                                       self.b_ffds[:-1],
                                                       self.a_attn_layer_norms[:-1],
                                                       self.b_attn_layer_norms[:-1],
                                                       self.a_ffd_layer_norms[:-1],
                                                       self.b_ffd_layer_norms[:-1])):
            attn_output, attn_weight = atn(
                q=torch.stack((
                    self.latent_queries_a[idx].unsqueeze(0).repeat(batch_size, 1, 1),
                    self.latent_queries_b[idx].unsqueeze(0).repeat(batch_size, 1, 1)
                ), dim=0).clone(),
                k=embedding.clone(),
                v=embedding.clone(),
                q_spa_embed=q_spa_embed,
                kv_spa_embed=ctx_spa_embed,
                mask=mask,
                attention_bias=gaussian_bias
            )
            attn_output = torch.stack((
                a_atn_ln(
                    self.latent_queries_a[idx].unsqueeze(0).repeat(batch_size, 1, 1) \
                    + attn_output[0]
                ),
                b_atn_ln(
                    self.latent_queries_b[idx].unsqueeze(0).repeat(batch_size, 1, 1) \
                    + attn_output[1]
                )
            ), dim=0)
            attn_output = torch.stack((
                a_ffd_ln(attn_output[0] + a_ffd(attn_output[0])),
                b_ffd_ln(attn_output[1] + b_ffd(attn_output[1]))
            ), dim=0)
            embedding = attn_output

        # Stand-alone
        if self.n_attn_layer != 1:
            mask = torch.zeros_like(mask, device=mask.device)
            gaussian_bias = torch.zeros_like(gaussian_bias, device=gaussian_bias.device)

        attn_output, attn_weight = self.attn_layers[-1](
            q=query.clone(),
            k=embedding.clone(),
            v=embedding.clone(),
            q_spa_embed=q_spa_embed,
            kv_spa_embed=ctx_spa_embed,
            mask=mask,
            attention_bias=gaussian_bias
        )  # -> [2, bs, ql(1), dm//2]
        # attn_weights = torch.cat((attn_weights, attn_weight))
        # residual connection valid when layer goes deep (YouTube stanford perceiver video)
        attn_output = torch.stack((
            self.a_attn_layer_norms[-1](query[0] + attn_output[0]),
            self.b_attn_layer_norms[-1](query[1] + attn_output[1])
        ), dim=0)  # -> [2, bs, ql(1), dm//2]
        attn_output = torch.stack((
            self.a_ffd_layer_norms[-1](attn_output[0] + self.a_ffds[-1](attn_output[0])),
            self.b_ffd_layer_norms[-1](attn_output[1] + self.b_ffds[-1](attn_output[1]))
        ), dim=0)  # -> [2, bs, ql(1), dm//2]

        return attn_output, attn_weights


class MaskedCartesianAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_a_head,
                 n_b_head,
                 attn_dropout=0.1,
                 attn_bias_factor=None):
        super(MaskedCartesianAttention, self).__init__()

        self.d_model = d_model
        self.n_a_head = n_a_head
        self.n_b_head = n_b_head
        self.n_head = self.n_a_head * self.n_b_head
        self.attn_bias_factor = attn_bias_factor

        self.w_q_a = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_q_b = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_k_a = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_k_b = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_v_a = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_v_b = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_o_a = nn.Linear(self.d_model // 2, self.d_model // 2, bias=False)
        self.w_o_b = nn.Linear(self.d_model // 2, self.d_model // 2, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(attn_dropout)

    def forward(self,
                q,
                k,
                v,
                q_spa_embed,
                kv_spa_embed,
                mask,
                attention_bias=None):
        """
        :param q: [2, bs, ql, dm//2]
        :param k: [2, bs, sl, dm//2]
        :param v: [2, bs, sl, dm//2]
        :param q_spa_embed: [bs, 1, dk, dk]
        :param kv_spa_embed: [bs, sl, dk, dk]
        :param mask: [bs, sl] element to mask => 1, valid => 0.
        :param attention_bias: [bs, nh, sl]

        Note that: n-head = nh = 4
        """
        _, batch_size, seq_len, _ = k.shape
        _, _, query_len, _ = q.shape

        # Project, Cross concat & Rotate
        # n-head = nh = 4
        # [bs, l, dm//2] -> [bs, l, dm//4]
        q_a = self.w_q_a(q[0])
        q_b = self.w_q_b(q[1])
        k_a = self.w_k_a(k[0])
        k_b = self.w_k_b(k[1])
        v_a = self.w_v_a(v[0])
        v_b = self.w_v_b(v[1])

        q_a = q_a.view(batch_size, query_len, 2, self.d_model // 8)
        q_b = q_b.view(batch_size, query_len, 2, self.d_model // 8)
        k_a = k_a.view(batch_size, seq_len, 2, self.d_model // 8)
        k_b = k_b.view(batch_size, seq_len, 2, self.d_model // 8)
        v_a = v_a.view(batch_size, seq_len, 2, self.d_model // 8)
        v_b = v_b.view(batch_size, seq_len, 2, self.d_model // 8)

        # Cartesian product
        # [bs, l, 2, dm//8] -> [bs, l, 4, dm//4] -> [bs, 4, l, dm//4]
        q = torch.cat((q_a.repeat(1, 1, 2, 1), q_b.repeat_interleave(2, dim=-2)), dim=-1).transpose(1, 2)
        k = torch.cat((k_a.repeat(1, 1, 2, 1), k_b.repeat_interleave(2, dim=-2)), dim=-1).transpose(1, 2)
        v = torch.cat((v_a.repeat(1, 1, 2, 1), v_b.repeat_interleave(2, dim=-2)), dim=-1).transpose(1, 2)

        # Add spatial embedding at FIRST layer (ll = ql = latent_len)
        if seq_len == kv_spa_embed.shape[1]:
            # Adapt spatial embeddings to match query and key dimensions
            # [bs, sl, dk, dk] -> [bs, 1, sl, dk, dk] -> [bs, 4, sl, dk//2, dk//2]
            q_spa_embed_reshaped = q_spa_embed.unsqueeze(1).repeat(1, 4, query_len, 1, 1)
            kv_spa_embed_reshaped = kv_spa_embed.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            
            # Ensure dimensions match for matrix multiplication
            q_dim = q.size(-1)
            spa_dim = q_spa_embed_reshaped.size(-1)
            
            if spa_dim != q_dim:
                # Reshape spatial embeddings or pad as needed
                if spa_dim > q_dim:
                    # Use a subset of the spatial dimensions
                    q_spa_embed_reshaped = q_spa_embed_reshaped[..., :q_dim, :q_dim]
                    kv_spa_embed_reshaped = kv_spa_embed_reshaped[..., :q_dim, :q_dim]
                else:
                    # Pad spatial dimensions (this should not happen with current setup)
                    pad_size = q_dim - spa_dim
                    q_spa_embed_reshaped = F.pad(q_spa_embed_reshaped, (0, pad_size, 0, pad_size))
                    kv_spa_embed_reshaped = F.pad(kv_spa_embed_reshaped, (0, pad_size, 0, pad_size))
            
            # Apply spatial embeddings to queries and keys
            # [bs, 4, ll, dm//4, dm//4] * [bs, 4, ll, dm//4, 1] -> [bs, 4, ll, dm//4]
            q = q_spa_embed_reshaped.matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, 4, sl, dm//4, dm//4] * [bs, 4, sl, dm//4, 1]
            k = kv_spa_embed_reshaped.matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at PROCESSOR layers:
        elif seq_len == query_len:
            # Adapt spatial embeddings to match query and key dimensions
            q_spa_embed_reshaped = q_spa_embed.unsqueeze(1).repeat(1, 4, query_len, 1, 1)
            
            # Ensure dimensions match
            q_dim = q.size(-1)
            spa_dim = q_spa_embed_reshaped.size(-1)
            
            if spa_dim != q_dim:
                if spa_dim > q_dim:
                    q_spa_embed_reshaped = q_spa_embed_reshaped[..., :q_dim, :q_dim]
                else:
                    pad_size = q_dim - spa_dim
                    q_spa_embed_reshaped = F.pad(q_spa_embed_reshaped, (0, pad_size, 0, pad_size))
            
            # Apply spatial embeddings
            q = q_spa_embed_reshaped.matmul(q.unsqueeze(-1)).squeeze(-1)
            k = q_spa_embed_reshaped.matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at DECODER layer (ql = 1)
        elif query_len == 1:
            # Adapt spatial embeddings to match query and key dimensions
            q_spa_embed_reshaped = q_spa_embed.unsqueeze(1).repeat(1, 4, query_len, 1, 1)
            
            # Ensure dimensions match
            q_dim = q.size(-1)
            spa_dim = q_spa_embed_reshaped.size(-1)
            
            if spa_dim != q_dim:
                if spa_dim > q_dim:
                    q_spa_embed_reshaped = q_spa_embed_reshaped[..., :q_dim, :q_dim]
                else:
                    pad_size = q_dim - spa_dim
                    q_spa_embed_reshaped = F.pad(q_spa_embed_reshaped, (0, pad_size, 0, pad_size))
            
            # Apply spatial embeddings
            q = q_spa_embed_reshaped.matmul(q.unsqueeze(-1)).squeeze(-1)
            k = q_spa_embed_reshaped.repeat(1, 1, seq_len, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)

        # [bs, nh, ql, dm//4] * [bs, nh, dm//4, sl] -> [bs, nh, ql, sl]
        attn_score = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.d_model // 4)  # dk

        if attention_bias.shape[-1] == seq_len:
            if attention_bias is None:
                raise NotImplementedError()
            else:
                # [bs, nh, sl] -> [bs, nh, ql, sl]
                attention_bias = attention_bias.unsqueeze(2).repeat(1, 1, query_len, 1)
        else:
            attention_bias = torch.zeros_like(attn_score, device=attn_score.device)

        if mask.shape[1] == seq_len:
            # [bs, nh, ql, sl] + [bs, nh, ql, sl] -> [bs, nh, ql, sl]
            attn_score += mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, query_len, 1) * -1e9

        attn_score = self.softmax(attn_score - attention_bias)
        attn_score = self.drop_out(attn_score)

        # [bs, nh, ql, l] * [bs, nh, l, dm//4] -> [bs, nh, ql, dm//4]
        v = attn_score.matmul(v)
        # [bs, nh, ql, dm//4] -> [bs, nh, ql, 2, dm//8]
        v = v.view(batch_size, self.n_head, query_len, 2, self.d_model // 8)

        v_a, v_b = v[:, :, :, 0], v[:, :, :, 1]
        v_a = v_a.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        v_b = v_b.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        v_a = self.w_o_a(v_a)  # -> [bs, ql, dm//2]
        v_b = self.w_o_b(v_b)  # -> [bs, ql, dm//2]
        v = torch.stack((v_a, v_b), dim=0)  # -> [2, bs, ql, dm//2]

        return v, attn_score


class VanillaPerceiver(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 n_attn_layer,
                 n_hidden_token=4,
                 attn_dropout=0.1,
                 attn_bias_factor=1):
        super(VanillaPerceiver, self).__init__()
        # ----------------------------------------------------------------
        self.d_model = d_model
        self.n_head = n_head
        self.n_attn_layer = n_attn_layer
        self.n_hidden_token = n_hidden_token
        self.attn_dropout = attn_dropout
        self.attn_bias_factor = attn_bias_factor
        # ----------------------------------------------------------------
        if attn_bias_factor:
            self.attn_bias_factor = attn_bias_factor
        else:
            self.attn_bias_factor = nn.Parameter(torch.zeros(1))

        self.attn_layers = nn.ModuleList([
            MaskedVanillaAttention(
                d_model=self.d_model,
                n_head=self.n_head,
                attn_dropout=self.attn_dropout,
                attn_bias_factor=self.attn_bias_factor
            )

            for _ in range(self.n_attn_layer)
        ])

        self.ffds = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.n_attn_layer)
        ])

        if self.n_attn_layer > 1:
            self.latent_queries = {
                n: nn.Parameter(torch.randn(self.n_hidden_token, self.d_model))
                for n in range(self.n_attn_layer - 1)
            }

        self.softmax = nn.Softmax(dim=3)

        self.attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])
        self.ffd_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])

    def forward(self,
                ctx_embed,
                q_embed,
                ctx_spa_embed,
                q_spa_embed,
                mask,
                geo_proxy):
        """
        :param ctx_embed: [bs, sl-1, dm]
        :param q_embed: [bs, sl-1, dm]
        :param ctx_spa_embed: [bs, sl-1, dm//4, dm//4]
        :param q_spa_embed: [bs, 1, dm//4, dm//4]
        :param mask: [bs, sl-1], masked:1, masked:0
        :param geo_proxy: [bs, sl-1]
        """
        batch_size = ctx_embed.shape[0]
        attn_weights = torch.FloatTensor([], device=ctx_embed.device)
        gaussian_bias = _calc_attention_bias(geo_proxy, self.attn_bias_factor)  # [bs, sl]
        embedding = ctx_embed

        # [bs, sl, dm], [bs, ql, dm]
        # Perceiver:
        for idx, (atn, ffd, atn_ln, ffd_ln) in enumerate(zip(
                self.attn_layers[:-1], self.ffds[:-1],
                self.attn_layer_norms[:-1], self.ffd_layer_norms[:-1]
        )):
            attn_output, attn_weight = atn(
                q=self.latent_queries[idx].unsqueeze(0).repeat(batch_size, 1, 1).clone(),  # -> [bs, ql, dm]
                k=ctx_embed.clone(),
                v=ctx_embed.clone(),
                q_spa_embed=q_spa_embed,
                kv_spa_embed=ctx_spa_embed,
                mask=mask,
                attention_bias=gaussian_bias
            )  # -> [bs, ql, dm]
            # residual connection:
            attn_output = atn_ln(
                self.latent_queries[idx] + attn_output
            )
            attn_output = ffd_ln(
                attn_output + ffd(attn_output)
            )
            embedding = attn_output

        # Stand-alone attention:
        if self.n_attn_layer != 1:
            mask = torch.zeros_like(mask, device=mask.device)
            gaussian_bias = torch.zeros_like(gaussian_bias, device=gaussian_bias.device)

        attn_output, attn_weight = self.attn_layers[-1](
            q=q_embed.clone(),
            k=embedding.clone(),
            v=embedding.clone(),
            q_spa_embed=q_spa_embed,
            kv_spa_embed=ctx_spa_embed,
            mask=mask,
            attention_bias=gaussian_bias
        )  # -> [bs, ql, dm]

        attn_output = self.attn_layer_norms[-1](
            q_embed + attn_output
        )
        attn_output = self.ffd_layer_norms[-1](
            attn_output + self.ffds[-1](attn_output)
        )  # -> [bs, ql, dm]

        return attn_output, attn_weights


class MaskedVanillaAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_head=4,
                 attn_dropout=0.1,
                 attn_bias_factor=1):
        super(MaskedVanillaAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.attn_bias_factor = attn_bias_factor

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(attn_dropout)

    def forward(self,
                q,
                k,
                v,
                q_spa_embed,
                kv_spa_embed,
                mask=None,
                attention_bias=None):
        """
        :param q: [bs, ql, dm]
        :param k: [bs, sl, dm]
        :param v: [bs, sl, dm]
        :param q_spa_embed: [bs, 1, dm//4, dm//4]
        :param kv_spa_embed: [bs, sl, dm//4, dm//4]
        :param mask: [bs, sl] element to mask => 1, valid => 0.
        :param attention_bias: [bs, nh, ql, sl]
        """
        batch_size, key_len, _ = k.shape
        _, query_len, _ = q.shape

        # Project
        # [bs, l, dm] -> [bs, l, dm]
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # [bs, l, dm] -> [bs, l, nh, dm//nh] -> [bs, nh, l, dm//nh]
        q = q.view(batch_size, query_len, self.n_head, self.d_model // self.n_head).transpose(1, 2)
        k = k.view(batch_size, key_len, self.n_head, self.d_model // self.n_head).transpose(1, 2)
        v = v.view(batch_size, key_len, self.n_head, self.d_model // self.n_head).transpose(1, 2)

        # Add spatial embedding at FIRST layer (ll = ql = latent_len)
        if key_len == kv_spa_embed.shape[1]:
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            q = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, 4, sl, dm//4, dm//4] * [bs, 4, sl, dm//4, 1] -> [bs, 4, sl, dm//4]
            k = kv_spa_embed.unsqueeze(1).repeat(1, self.n_head, 1, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at PROCESSOR layers:
        elif key_len == query_len:
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            q = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            k = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at DECODER layer (ql = 1)
        elif query_len == 1:
            # [bs, nh, ql, dm//4, dm//4] * [bs, nh, ql, dm//4, 1] -> [bs, nh, ql, dm//4]
            q = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            k = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, key_len, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)

        # [bs, nh, ql, dm//4] * [bs, nh, dm//4, sl] -> [bs, nh, ql, sl]
        attn_score = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.d_model // self.n_head)  # dk

        if attention_bias.shape[1] == key_len:
            if attention_bias is None:
                raise NotImplementedError()
            else:
                # [bs, sl] -> [bs, nh, ql, sl]
                attention_bias = attention_bias.unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, query_len, 1)
        else:
            attention_bias = torch.zeros_like(attn_score, device=attn_score.device)

        if mask.shape[1] == key_len:
            # [bs, nh, ql, sl] + [bs, nh, ql, sl] -> [bs, nh, ql, sl]
            attn_score += mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, query_len, 1) * -1e9

        attn_score = self.softmax(attn_score - attention_bias)
        attn_score = self.drop_out(attn_score)

        # [bs, nh, ql, l] * [bs, nh, l, dm//nh] -> [bs, nh, ql, dm//nh] -> [bs, ql, nh, dm//nh] -> [bs, ql, dm]
        v = attn_score.matmul(v).transpose(2, 3).contiguous().view(batch_size, query_len, self.d_model)
        v = self.w_o(v)  # -> [bs, ql, dm]

        return v, attn_score
