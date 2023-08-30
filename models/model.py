import numbers

import einops
from einops import rearrange

from models.backbone import *
from models.blocks import *


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


class LapPyramidConv(nn.Module):
    def __init__(self, num_high=4):
        super(LapPyramidConv, self).__init__()

        self.num_high = num_high
        self.kernel = gauss_kernel()

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


# ----------- Texture Recovery Module -------------------
class TRM(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(TRM, self).__init__()

        self.num_high = num_high

        blocks = [nn.Conv2d(9, 64, 3, padding=1),
                  nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            blocks += [ResidualBlock(64)]

        blocks += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model = nn.Sequential(*blocks)

        channels = 3
        # Stage1
        self.block1_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=2,
                                  norm=None, nonlinear='leakyrelu')
        self.block1_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=4,
                                  norm=None, nonlinear='leakyrelu')
        self.aggreation1_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        # Stage2
        self.block2_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=8,
                                  norm=None, nonlinear='leakyrelu')
        self.block2_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=16,
                                  norm=None, nonlinear='leakyrelu')
        self.aggreation2_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        # Stage3
        self.block3_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=32,
                                  norm=None, nonlinear='leakyrelu')
        self.block3_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=64,
                                  norm=None, nonlinear='leakyrelu')
        self.aggreation3_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        # Stage3
        self.spp_img = SPP(in_channels=channels, out_channels=channels, num_layers=4, interpolation_type='bicubic')
        self.block4_1 = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x, pyr_original, fake_low):
        pyr_result = [fake_low]
        mask = self.model(x)

        mask = nn.functional.interpolate(mask, size=(pyr_original[-2].shape[2], pyr_original[-2].shape[3]))
        result_highfreq = torch.mul(pyr_original[-2], mask) + pyr_original[-2]
        out1_1 = self.block1_1(result_highfreq)
        out1_2 = self.block1_2(out1_1)
        agg1_rgb = self.aggreation1_rgb(torch.cat((result_highfreq, out1_1, out1_2), dim=1))
        pyr_result.append(agg1_rgb)

        mask = nn.functional.interpolate(mask, size=(pyr_original[-3].shape[2], pyr_original[-3].shape[3]))
        result_highfreq = torch.mul(pyr_original[-3], mask) + pyr_original[-3]
        out2_1 = self.block2_1(result_highfreq)
        out2_2 = self.block2_2(out2_1)
        agg2_rgb = self.aggreation2_rgb(torch.cat((result_highfreq, out2_1, out2_2), dim=1))

        out3_1 = self.block3_1(agg2_rgb)
        out3_2 = self.block3_2(out3_1)
        agg3_rgb = self.aggreation3_rgb(torch.cat((agg2_rgb, out3_1, out3_2), dim=1))

        spp_rgb = self.spp_img(agg3_rgb)
        out_rgb = self.block4_1(spp_rgb)

        pyr_result.append(out_rgb)
        pyr_result.reverse()

        return pyr_result


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Axis-based Multi-head Self-Attention

class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


# Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)

        return x


# Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Axis-based Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Tri-layer Attention Alignment Block
class TAA(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim, bias=True):
        super(TAA, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        return out


##########################################################################
# ---------- Dimension-Aware Transformer Block -----------------------
class Backbone(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=3,
                 num_blocks=[1, 2, 4, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 attention=True,
                 ):
        super(Backbone, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.encoder_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.encoder_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.layer_fussion = TAA(in_dim=int(dim * 3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.trans_low = DFE()

        self.coefficient_1_0 = nn.Parameter(torch.ones((2, int(int(dim)))), requires_grad=attention)

        self.refinement_1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.refinement_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.refinement_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])

        self.layer_fussion_2 = TAA(in_dim=int(dim * 3))
        self.conv_fuss_2 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        inp_enc_encoder1 = self.patch_embed(inp)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)

        inp_fusion_123 = torch.cat(
            [out_enc_encoder1.unsqueeze(1), out_enc_encoder2.unsqueeze(1), out_enc_encoder3.unsqueeze(1)], dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        out_enc = self.trans_low(out_fusion_123)

        out_fusion_123 = self.latent(out_fusion_123)

        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123 + self.coefficient_1_0[1, :][None, :,
                                                                                 None, None] * out_enc

        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)

        inp_fusion = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)
        result = self.output(out)

        return result


class Model(nn.Module):
    def __init__(self, depth=2):
        super(Model, self).__init__()
        self.backbone = Backbone()
        self.lap_pyramid = LapPyramidConv(depth)
        self.trans_high = TRM(3, num_high=depth)

    def forward(self, inp):
        pyr_inp = self.lap_pyramid.pyramid_decom(img=inp)
        out_low = self.backbone(pyr_inp[-1])

        inp_up = nn.functional.interpolate(pyr_inp[-1], size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        out_up = nn.functional.interpolate(out_low, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        high_with_low = torch.cat([pyr_inp[-2], inp_up, out_up], 1)

        pyr_inp_trans = self.trans_high(high_with_low, pyr_inp, out_low)

        result = self.lap_pyramid.pyramid_recons(pyr_inp_trans)

        return result


if __name__ == '__main__':
    # tensor = torch.randn(1, 3, 1024, 1024).cuda()
    # model = Model().cuda()
    # output = model(tensor)
    # print(output.shape)
    from thop import profile, clever_format

    model = Model()
    input = torch.randn(1, 3, 256, 256)
    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(params)
