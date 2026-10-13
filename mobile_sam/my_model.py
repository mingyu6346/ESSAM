import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobile_sam import sam_model_registry

# Pretrained MobileSAM backbone shipped with this release:
#   <repo root>/weights/mobile_sam.pt
_DEFAULT_CKPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "mobile_sam.pt")
)


def build_model(device="cuda", img_size=384, in_chans=3, checkpoint=None):
    """Build the MobileSAM (vit_t) backbone and load the pretrained weights.

    Args:
        device: torch device for the backbone.
        img_size: input resolution the SAM decoder works at.
        in_chans: input image channels.
        checkpoint: path to the MobileSAM pretrained weights. Defaults to
            `<repo root>/weights/mobile_sam.pt`.
    """
    model_type = "vit_t"
    sam_checkpoint = checkpoint or _DEFAULT_CKPT
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint, img_size=img_size, in_chans=in_chans,
                                           pretrain_mode=None)
    model.to(device)
    return model


class Model(nn.Module):
    def __init__(self, img_size=384, in_chans=3):
        super().__init__()
        self.sam = build_model(img_size=img_size, in_chans=in_chans)
        self.mpa = MPA()
        self.landslide_head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        image_embeddings, f_ls = self.sam.image_encoder(x)

        f_agg = self.mpa(f_ls)
        landslide_mask = self.landslide_head(f_agg)

        mask_input = torch.sigmoid(landslide_mask)

        batch_size = x.size(0)
        final_masks = []
        for b in range(batch_size):
            img_feat = image_embeddings[b].unsqueeze(0)
            mask_prompt = mask_input[b].unsqueeze(0)

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=None, masks=mask_prompt)
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=img_feat,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            final_mask = self.sam.postprocess_masks(low_res_masks, input_size=x.shape[-2:],
                                                    original_size=x.shape[-2:])
            final_masks.append(final_mask)

        final_mask = torch.cat(final_masks, dim=0)
        landslide_mask = F.interpolate(landslide_mask,size=final_mask.shape[2:],mode='bilinear')

        return landslide_mask, final_mask


class MPA(nn.Module):
    def __init__(self, channel_ls=[64, 160, 320, 320], hw_ls=[128, 32, 32, 32]):
        super(MPA, self).__init__()
        self.ppa_3 = PPA(high_dim=channel_ls[3], low_dim=channel_ls[2], kernel_size=3,
                         reduction=9, ghost_dw_size=3, ghost_dw_size2=5, use_upsample=False)
        self.ppa_2 = PPA(high_dim=channel_ls[2], low_dim=channel_ls[1], kernel_size=5,
                         reduction=8, ghost_dw_size=3, ghost_dw_size2=7, use_upsample=False)
        self.ppa_1 = PPA(high_dim=channel_ls[1], low_dim=channel_ls[0], kernel_size=7,
                         reduction=4, ghost_dw_size=5, ghost_dw_size2=7, use_upsample=True)

    def forward(self, fea_ls):
        xf_3 = self.ppa_3(fea_ls[2], fea_ls[3])
        xf_2 = self.ppa_2(fea_ls[1], xf_3)
        xf_1 = self.ppa_1(fea_ls[0], xf_2)

        return xf_1



class PPA(nn.Module):

    def __init__(self, high_dim, low_dim, kernel_size=3, use_upsample=False,
                 reduction=4, ghost_dw_size=3, ghost_dw_size2=3):
        super().__init__()

        self.use_upsample = use_upsample

        self.ghost_reduce = GhostModule(high_dim, low_dim, ratio=2, dw_size=ghost_dw_size)
        self.ghost_reduce2 = GhostModule(low_dim, low_dim, ratio=2, dw_size=ghost_dw_size2)
        self.dynamic_conv = DynamicDepthwiseConv(low_dim * 2, kernel_size, reduction=reduction)

        self.eca1 = EfficientChannelAttention(low_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(low_dim * 2, low_dim, 1, bias=False),
            nn.BatchNorm2d(low_dim),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def forward(self, x_low, x_high):
        B, C, H, W = x_low.shape
        # 上采样高层特征
        x_high = F.interpolate(x_high, size=(H, W), mode="bilinear",
                               align_corners=True) if self.use_upsample else x_high

        # Ghost模块降维
        fused = torch.cat([self.ghost_reduce(x_high), self.ghost_reduce2(x_low)], dim=1)

        # avg+dbg
        fused = self.dynamic_conv(fused)

        # 特征融合
        fused = self.fusion(fused)

        # 特征拼接
        fused = fused + x_low

        # 第一次注意力增强
        fused = fused * self.eca1(fused)
        return fused

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class EfficientChannelAttention(nn.Module):
    """轻量级ECA注意力机制，使用1D卷积替代FC层"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应计算卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class DynamicDepthwiseConv(nn.Module):
    """动态深度卷积，根据输入自适应调整"""

    def __init__(self, channels, kernel_size=5, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.dwconv = nn.Conv2d(channels, channels, kernel_size,
                                padding=kernel_size // 2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        att = self.fc(self.avg_pool(x))
        out = self.dwconv(x)
        out = self.bn(out)
        return out * att


class GhostModule(nn.Module):
    """Ghost模块，用更少的参数生成更多特征图"""

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.GELU()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class ExpendConv(nn.Module):
    def __init__(self, in_channels, out_channels, extend_ratio=2):
        super(ExpendConv, self).__init__()
        extend_channels = extend_ratio * in_channels
        self.conv_1 = nn.Conv2d(in_channels, extend_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(extend_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv_2(self.act(self.conv_1(x)))
