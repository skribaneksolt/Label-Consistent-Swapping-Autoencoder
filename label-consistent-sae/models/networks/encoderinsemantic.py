import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import util
from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ResBlock, ConvLayer, ToRGB, EqualLinear, Blur, Upsample, make_kernel
from models.networks.stylegan2_op import upfirdn2d


class Upsample2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        demodulate=True,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        factor = 2
        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1

        self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.demodulate = demodulate
        self.new_demodulation = True

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        weight = self.scale * self.weight
        
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        out = self.blur(out)

        return out


class StyleGAN2ResnetEncoderInSemantic(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
        parser.add_argument("--netE_num_downsampling_sp", default=4, type=int)
        parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
        parser.add_argument("--netE_nc_steepness", default=2.0, type=float)
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if self.opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(self.opt.netE_num_downsampling_sp)  # = 512 by default
        self.add_module("ToSpatialCode1", ConvLayer(nchannels, nchannels, 1, activate=True, bias=True))
        self.add_module("ToSpatialCode2",
                        ConvLayer(nchannels, self.opt.spatial_code_ch, kernel_size=1, activate=False, bias=True))
        
        self.add_module(
            "ToInnerSemantic",
            nn.Sequential(
                nn.ConvTranspose2d(nchannels, nchannels // 2, kernel_size=2, stride=2),
                nn.ReLU(),
                ConvLayer(nchannels//2, nchannels//2, 3, blur_kernel=[1], downsample=False, pad=1),
                nn.ConvTranspose2d(nchannels//2, nchannels//4, kernel_size=2, stride=2),
                nn.ReLU(),
                ConvLayer(nchannels//4, nchannels//4, 3, blur_kernel=[1], downsample=False, pad=1),
                ConvLayer(nchannels//4, 18, 1, blur_kernel=[1], downsample=False, pad=0, activate=False)
            )
        )
        
        self.DownToGlobalCode = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_gl):
            idx_from_beginning = self.opt.netE_num_downsampling_sp + i
            self.DownToGlobalCode.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=0)
            )

        nchannels = self.nc(self.opt.netE_num_downsampling_sp +     # SOLT : it means 2^11 = 2048
                            self.opt.netE_num_downsampling_gl)
        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(nchannels, self.opt.global_code_ch)
            )
        )

    def nc(self, idx):
        nc = self.opt.netE_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE_scale_capacity
        # nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)
        
        
    # def forward(self, x, extract_features=False):
    def forward(self, x, extract_inner_sem=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp1 = self.ToSpatialCode1(midpoint)
        sp = self.ToSpatialCode2(sp1)
        inner_sem = self.ToInnerSemantic(sp1)
        # sp = self.ToSpatialCode(midpoint)

        # if extract_features:
        #     padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
        #     feature = self.DownToGlobalCode[0](padded_midpoint)
        #     assert feature.size(2) == sp.size(2) // 2 and \
        #         feature.size(3) == sp.size(3) // 2
        #     feature = F.interpolate(
        #         feature, size=(7, 7), mode='bilinear', align_corners=False)

        x = self.DownToGlobalCode(midpoint)
        x = x.mean(dim=(2, 3))
        gl = self.ToGlobalCode(x)
        sp = util.normalize(sp)
        gl = util.normalize(gl)
        # if extract_features:
        #     return sp, gl, feature
        if extract_inner_sem:
            return sp, gl, inner_sem
        else:
            return sp, gl
