"""
Modules for model components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from transformers import CLIPTokenizer, CLIPTextModel


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False,
        t_emb=None,
    ) -> None:
        super().__init__()
        """
        Standard ResNet style convolutional block
        """
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if t_emb is not None:
            self.time = nn.Sequential(
                nn.Linear(in_features=t_emb, out_features=out_channels),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor, t_emb=None) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            if t_emb is not None:
                x1 += self.time(t_emb)[:, :, None, None]
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out
        else:
            x1 = self.conv1(x)
            if t_emb is not None:
                x1 += self.time(t_emb)[:, :, None, None]
            x2 = self.conv2(x1)
            return x2


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, context=None):
        return self.fn(x, context) + x


class RearrangeToSequence(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, context=None):
        x = rearrange(x, "b c ... -> b ... c")
        x, ps = pack([x], "b * c")

        x = self.fn(x, context)

        (x,) = unpack(x, ps, "b * c")
        x = rearrange(x, "b ... c -> b c ...")
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        rotary_emb=None,
        cosine_sim=True,
        cosine_sim_scale=16
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        # rotary embeddings
        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net
        nk, nv = map(
            lambda t: repeat(t, "d -> b 1 d", b=b), self.null_kv.unbind(dim=-2)
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # whether to use cosine sim
        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities
        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(
                (i, j), dtype=torch.bool, device=device
            ).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values
        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """General implementation of Cross & Self Attention multi-head"""

    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=8):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            # Self Attention
            self.to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.self_attn = True
        else:
            # Cross Attention
            self.to_k = nn.Linear(context_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(context_dim, embed_dim, bias=False)
            self.self_attn = False
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True)
        )  # this could be omitted

    def forward(self, tokens, context=None):
        Q = self.to_q(tokens)
        K = self.to_k(tokens) if self.self_attn else self.to_k(context)
        V = self.to_v(tokens) if self.self_attn else self.to_v(context)
        # print(Q.shape, K.shape, V.shape)
        # transform heads onto batch dimension
        Q = rearrange(
            Q, "B T (H D) -> (B H) T D", H=self.num_heads, D=self.head_dim
        )
        K = rearrange(
            K, "B T (H D) -> (B H) T D", H=self.num_heads, D=self.head_dim
        )
        V = rearrange(
            V, "B T (H D) -> (B H) T D", H=self.num_heads, D=self.head_dim
        )
        # print(Q.shape, K.shape, V.shape)
        scoremats = torch.einsum("BTD,BSD->BTS", Q, K)
        attnmats = F.softmax(scoremats / math.sqrt(self.head_dim), dim=-1)
        # print(scoremats.shape, attnmats.shape, )
        ctx_vecs = torch.einsum("BTS,BSD->BTD", attnmats, V)
        # split the heads transform back to hidden.
        ctx_vecs = rearrange(
            ctx_vecs,
            "(B H) T D -> B T (H D)",
            H=self.num_heads,
            D=self.head_dim,
        )
        # TODO: note this `to_out` is also a linear layer, could be in principle merged into the to_value layer.
        return self.to_out(ctx_vecs)


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, type_attention, text):
        super(UnetDown, self).__init__()
        """
        Process and downscale the image feature maps
        """
        self.text = text
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        ]
        attention = nn.Identity()
        if type_attention == "self":
            create_self_attn = lambda dim: RearrangeToSequence(
                Residual(Attention(dim))
            )
            attention = create_self_attn(out_channels)
        if type_attention == "cross" or self.text:
            create_self_attn = lambda dim: RearrangeToSequence(
                Residual(CrossAttention(dim, dim, 768 if self.text else None))
            )
            attention = create_self_attn(out_channels)
        if self.text:
            self.resblock = ResidualConvBlock(
                in_channels, out_channels, t_emb=768
            )
            self.attn = attention
            self.pool = nn.MaxPool2d(2)
        else:
            self.model = nn.Sequential(
                *[
                    ResidualConvBlock(in_channels, out_channels),
                    attention,
                    nn.MaxPool2d(2),
                ]
            )

    def forward(self, x, t_emb=None, c_emb=None):
        return (
            self.pool(self.attn(self.resblock(x, t_emb), c_emb))
            if self.text
            else self.model(x)
        )


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, type_attention, text):
        super(UnetUp, self).__init__()
        """
        process and upscale the image feature maps
        """
        self.text = text
        attention = nn.Identity()
        if type_attention == "self":
            create_self_attn = lambda dim: RearrangeToSequence(
                Residual(Attention(dim))
            )
            attention = create_self_attn(out_channels)
        if type_attention == "cross" or self.text:
            create_self_attn = lambda dim: RearrangeToSequence(
                Residual(CrossAttention(dim, dim, 768 if self.text else None))
            )
            attention = create_self_attn(out_channels)
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(
                out_channels, out_channels, t_emb=768 if self.text else None
            ),
            attention,
            ResidualConvBlock(
                out_channels, out_channels, t_emb=768 if self.text else None
            ),
        ]
        if self.text:
            self.layers = nn.ModuleList(layers)
        else:
            self.model = nn.Sequential(*layers)

    def forward(self, x, skip, t_emb=None, c_emb=None):
        x = torch.cat((x, skip), 1)
        if self.text:
            return self.layers[-1](
                self.layers[2](
                    self.layers[1](self.layers[0](x), t_emb), c_emb
                ),
                t_emb,
            )
        else:
            return self.model(x)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        Generic one layer FC NN for embedding things
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(
        self,
        text,
        in_channels,
        n_feat=256,
        n_classes=10,
        dataset="",
        type_attention="",
        pixel_size=28,
        device=None,
    ):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_contexts = len(n_classes)
        self.n_feat = 2 * n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.text = text
        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.down1 = UnetDown(n_feat, n_feat, type_attention, self.text)
        self.down2 = UnetDown(n_feat, 2 * n_feat, type_attention, self.text)

        self.up1 = UnetUp(4 * n_feat, n_feat, type_attention, self.text)
        self.up2 = UnetUp(2 * n_feat, n_feat, type_attention, self.text)

        self.to_vec = nn.Sequential(nn.AvgPool2d(pixel_size // 4), nn.GELU())

        ### embedding shape
        self.dataset = dataset
        self.n_out1 = 2 * n_feat
        self.n_out2 = n_feat
        if text:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.transformer = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).eval()
            self.max_length = 77
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.timeembed = EmbedFC(1, 768)
        else:
            self.timeembed1 = EmbedFC(1, 2 * n_feat)
            self.timeembed2 = EmbedFC(1, 1 * n_feat)
            self.contextembed1 = nn.ModuleList(
                [
                    EmbedFC(self.n_classes[iclass], self.n_out1)
                    #nn.Linear(self.n_classes[iclass], self.n_out1, bias=False)
                    for iclass in range(len(self.n_classes))
                ]
            )
            self.contextembed2 = nn.ModuleList(
                [
                    EmbedFC(self.n_classes[iclass], self.n_out2)
                    #nn.Linear(self.n_classes[iclass], self.n_out2, bias=False)
                    for iclass in range(len(self.n_classes))
                ]
            )

        n_conv = pixel_size // 4
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, n_conv, n_conv),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, context_label, t, context_mask=None):
        # x is (noisy) image, c is context label, t is timestep,
        # x: [batch, channels, width(?), height(?)]
        x = self.init_conv(x)  # [batch, n_feat, width, height]
        if self.text:
            batch_encoding = self.tokenizer(
                context_label,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(self.device)
            outputs = self.transformer(input_ids=tokens)
            cemb = outputs.last_hidden_state
            temb = self.timeembed(t)

        else:
            temb1 = self.timeembed1(t).view(-1, int(self.n_feat), 1, 1)
            temb2 = self.timeembed2(t).view(-1, int(self.n_feat / 2), 1, 1)

            cemb1 = 0
            cemb2 = 0
            for ic in range(len(self.n_classes)):
                tmpc = context_label[ic]
                if tmpc.dtype == torch.int64:
                    tmpc = nn.functional.one_hot(
                        tmpc, num_classes=self.n_classes[ic]
                    ).type(torch.float)
                cemb1 += self.contextembed1[ic](tmpc).view(
                    -1, int(self.n_out1 / 1.0), 1, 1
                )
                cemb2 += self.contextembed2[ic](tmpc).view(
                    -1, int(self.n_out2 / 1.0), 1, 1
                )

        # [b, f, w/2, h/2]
        down1 = self.down1(x, temb, cemb) if self.text else self.down1(x)

        # [b, 2*f, w/4, h/4]
        down2 = (
            self.down2(down1, temb, cemb) if self.text else self.down2(down1)
        )

        # [b, 2*f, 1, 1]
        hiddenvec = self.to_vec(down2)

        # [b, 2*f, w/4, h/4]
        up1 = self.up0(hiddenvec)

        # [b, f, w/2, h/2]
        up2 = (
            self.up1(up1, down2, temb, cemb)
            if self.text
            else self.up1(cemb1 * up1 + temb1, down2)
        )

        # [b, f, w, h]
        up3 = (
            self.up2(up2, down1, temb, cemb)
            if self.text
            else self.up2(cemb2 * up2 + temb2, down1)
        )
        # [b, channel, w, h]
        out = self.out(torch.cat((up3, x), 1))
        return out
