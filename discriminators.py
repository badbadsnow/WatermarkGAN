# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.
from random import sample

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Spectrogram, Resample
import julius
import typing
from typing import Optional, List, Union, Dict, Tuple

from env import AttrDict
from utils import get_padding
from seanet.seanet import SEANetEncoderKeepDimension


class DiscriminatorP(torch.nn.Module):
    def __init__(
            self,
            h: AttrDict,
            period: List[int],
            kernel_size: int = 5,
            stride: int = 3,
            use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        int(32 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(32 * self.d_mult),
                        int(128 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(128 * self.d_mult),
                        int(512 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(512 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(1024 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        1,
                        padding=(2, 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h: AttrDict):
        super().__init__()
        self.mpd_reshapes = h.mpd_reshapes
        print(f"mpd_reshapes: {self.mpd_reshapes}")
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm)
                for rs in self.mpd_reshapes
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[List[int]]):
        super().__init__()

        self.resolution = resolution
        assert (
                len(self.resolution) == 3
        ), f"MRD layer requires list with len=3, got {self.resolution}"
        self.lrelu_slope = 0.1

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, "mrd_use_spectral_norm"):
            print(
                f"[INFO] overriding MRD use_spectral_norm as {cfg.mrd_use_spectral_norm}"
            )
            norm_f = (
                weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
            )
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, "mrd_channel_mult"):
            print(f"[INFO] overriding mrd channel multiplier as {cfg.mrd_channel_mult}")
            self.d_mult = cfg.mrd_channel_mult

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        # 5 层self.convs + 1层postconv
        # [torch.Size([1, 32, 513, 68]), torch.Size([1, 32, 513, 34]), torch.Size([1, 32, 513, 17]), torch.Size([1, 32, 513, 9]), torch.Size([1, 32, 513, 9]), torch.Size([1, 1, 513, 9])]
        # x = torch.Size([1, 4617])
        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert (
                len(self.resolutions) == 3
        ), f"MRD requires list of list with len=3, each element having a list with len=3. Got {self.resolutions}"
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Method based on descript-audio-codec: https://github.com/descriptinc/descript-audio-codec
# Modified code adapted from https://github.com/gemelo-ai/vocos under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorB(nn.Module):
    def __init__(
            self,
            window_length: int,
            channels: int = 32,
            hop_factor: float = 0.25,
            bands: Tuple[Tuple[float, float], ...] = (
                    (0.0, 0.1),
                    (0.1, 0.25),
                    (0.25, 0.5),
                    (0.5, 0.75),
                    (0.75, 1.0),
            ),
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length,
            hop_length=int(window_length * hop_factor),
            win_length=window_length,
            power=None,
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))
                ),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        self.conv_post = weight_norm(
            nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1))
        )

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 2, 1)  # [B, F, T, C] -> [B, C, T, F]
        # Split into bands
        x_bands = [x[..., b[0]: b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        x = []

        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


# Method based on descript-audio-codec: https://github.com/descriptinc/descript-audio-codec
# Modified code adapted from https://github.com/gemelo-ai/vocos under the MIT license.
#   LICENSE is in incl_licenses directory.
class MultiBandDiscriminator(nn.Module):
    def __init__(
            self,
            h,
    ):
        """
        Multi-band multi-scale STFT discriminator, with the architecture based on https://github.com/descriptinc/descript-audio-codec.
        and the modified code adapted from https://github.com/gemelo-ai/vocos.
        """
        super().__init__()
        # fft_sizes (list[int]): Tuple of window lengths for FFT. Defaults to [2048, 1024, 512] if not set in h.
        self.fft_sizes = h.get("mbd_fft_sizes", [2048, 1024, 512])
        self.discriminators = nn.ModuleList(
            [DiscriminatorB(window_length=w) for w in self.fft_sizes]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg: AttrDict, hop_length: int, n_octaves: int, bins_per_octave: int):
        super().__init__()
        self.cfg = cfg

        self.filters = cfg["cqtd_filters"]
        self.max_filters = cfg["cqtd_max_filters"]
        self.filters_scale = cfg["cqtd_filters_scale"]
        self.kernel_size = (3, 9)
        self.dilations = cfg["cqtd_dilations"]
        self.stride = (1, 2)

        self.in_channels = cfg["cqtd_in_channels"]
        self.out_channels = cfg["cqtd_out_channels"]
        self.fs = cfg["sampling_rate"]
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = self.cfg.get("cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print(
                f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def get_2d_padding(
            self,
            kernel_size: typing.Tuple[int, int],
            dilation: typing.Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                    :,
                    :,
                    :,
                    i * self.bins_per_octave: (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()

        self.cfg = cfg
        # Using get with defaults
        self.cfg["cqtd_filters"] = self.cfg.get("cqtd_filters", 32)
        self.cfg["cqtd_max_filters"] = self.cfg.get("cqtd_max_filters", 1024)
        self.cfg["cqtd_filters_scale"] = self.cfg.get("cqtd_filters_scale", 1)
        self.cfg["cqtd_dilations"] = self.cfg.get("cqtd_dilations", [1, 2, 4])
        self.cfg["cqtd_in_channels"] = self.cfg.get("cqtd_in_channels", 1)
        self.cfg["cqtd_out_channels"] = self.cfg.get("cqtd_out_channels", 1)
        # Multi-scale params to loop over
        self.cfg["cqtd_hop_lengths"] = self.cfg.get("cqtd_hop_lengths", [512, 256, 256])
        self.cfg["cqtd_n_octaves"] = self.cfg.get("cqtd_n_octaves", [9, 9, 9])
        self.cfg["cqtd_bins_per_octaves"] = self.cfg.get(
            "cqtd_bins_per_octaves", [24, 36, 48]
        )

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    self.cfg,
                    hop_length=self.cfg["cqtd_hop_lengths"][i],
                    n_octaves=self.cfg["cqtd_n_octaves"][i],
                    bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
                )
                for i in range(len(self.cfg["cqtd_hop_lengths"]))
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CombinedDiscriminator(nn.Module):
    """
    Wrapper of chaining multiple discrimiantor architectures.
    Example: combine mbd and cqtd as a single class
    """

    def __init__(self, list_discriminator: List[nn.Module]):
        super().__init__()
        self.discrimiantor = nn.ModuleList(list_discriminator)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discrimiantor:
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            y_d_rs.extend(y_d_r)
            fmap_rs.extend(fmap_r)
            y_d_gs.extend(y_d_g)
            fmap_gs.extend(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WatermarkDetector(nn.Module):
    def __init__(self, h: AttrDict):
        super().__init__()
        self.sr = h.get("sampling_rate", 16000)  # 音频采样率
        self.watermark_dim = h.get("watermark_dim", 32)
        self.win_sec = h.get("win_sec", 0.1)  # 基础卷积窗口时长(秒)
        self.feat_dim = h.get("feat_dim", 128)  # 特征向量维度
        self.scale_factors = h.get("scale_factors", [0.5, 1.0, 2.0])  # 多尺度卷积的窗口缩放因子
        self.conv_channels = h.get("conv_channels", 32)  # 卷积层通道数
        self.attn_dim = h.get("attn_dim", 256)  # 注意力机制隐藏层维度
        self.dropout_rate = h.get("dropout_rate", 0.3)  # 解码器Dropout比率
        self._build_layers()

    def _build_layers(self):
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.conv_channels,
                          kernel_size=int(self.sr * self.win_sec * scale),
                          stride=int(self.sr * self.win_sec * 0.1),  # 减小步长保留更多特征
                          padding=int(self.sr * self.win_sec * scale // 2)),  # 动态填充保持尺寸
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool1d(16)  # 自适应池化压缩时间维度至固定长度
            ) for scale in self.scale_factors
        ])

        self.feature_compress = nn.ModuleList([
            nn.Linear(16 * self.conv_channels, self.feat_dim)  # [32 * 16] -> 128
            for _ in self.scale_factors
        ])

        self.attention = nn.Sequential(
            nn.Linear(len(self.scale_factors) * self.feat_dim, self.attn_dim),
            nn.Tanh(),
            nn.Linear(self.attn_dim, len(self.scale_factors)),
            nn.Softmax(dim=-1)
        )

        self.decoder_exist = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 1),  # 水印存在概率
            nn.Sigmoid()
        )

        self.decoder_content = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.watermark_dim),  # 水印内容预测
            nn.Sigmoid()
        )

    def forward(self, x):
        """ x: 音频波形 [batch, 1, samples] """
        features = []
        for i, conv in enumerate(self.conv_layers):
            feat = conv(x)
            feat = feat.view(feat.size(0), -1)  # 展平时间与通道维度
            feat = self.feature_compress[i](feat)
            features.append(feat)

        # 多尺度特征拼接: [1, 128 * 3] = [1, 384]
        multi_scale_feat = torch.cat(features, dim=-1)

        # 后续注意力与解码逻辑保持不变
        attn_weights = self.attention(multi_scale_feat)

        weighted_feat = torch.stack([
            feat * attn_weights[:, i].unsqueeze(1)
            for i, feat in enumerate(features)
        ], dim=0).sum(dim=0)  # [batch, feat_dim]

        prob_exist = self.decoder_exist(weighted_feat)
        content_pred = self.decoder_content(weighted_feat)
        return prob_exist, content_pred


class AudioSealDetector(torch.nn.Module):
    """
    Detect the watermarking from an audio signal
    Args:
        SEANetEncoderKeepDimension (_type_): _description_
        nbits (int): The number of bits in the secret message. The result will have size
            of 2 + nbits, where the first two items indicate the possibilities of the
            audio being watermarked (positive / negative scores), he rest is used to decode
            the secret message. In 0bit watermarking (no secret message), the detector just
            returns 2 values.
    """

    def __init__(self, h: AttrDict):
        super().__init__()
        self.seanet_configs = h.get("seanet")
        self.sampling_rate = h.get("sampling_rate", 24000)
        self.nbits = h.get("nbits", 16)
        self.output_dim = h.get("output_dim", 32)
        encoder = SEANetEncoderKeepDimension(
            activation=self.seanet_configs.get("activation"),
            activation_params=self.seanet_configs.get("activation_params"),
            causal=self.seanet_configs.get("causal"),
            # channels=self.seanet_configs.get("channels"),
            channels=1,
            compress=self.seanet_configs.get("compress"),
            dilation_base=self.seanet_configs.get("dilation_base"),
            dimension=self.seanet_configs.get("dimension"),
            disable_norm_outer_blocks=self.seanet_configs.get(
                "disable_norm_outer_blocks"),
            kernel_size=self.seanet_configs.get("kernel_size"),
            last_kernel_size=self.seanet_configs.get("last_kernel_size"),
            lstm=self.seanet_configs.get("lstm"),
            n_filters=self.seanet_configs.get("n_filters"),
            n_residual_layers=self.seanet_configs.get("n_residual_layers"),
            norm=self.seanet_configs.get("norm"),
            norm_params=self.seanet_configs.get("norm_params"),
            pad_mode=self.seanet_configs.get("pad_mode"),
            ratios=self.seanet_configs.get("ratios"),
            residual_kernel_size=self.seanet_configs.get("residual_kernel_size"),
            true_skip=self.seanet_configs.get("true_skip"),
            output_dim=self.output_dim,
        )
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + self.nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)

    def detect_watermark(
            self,
            x: torch.Tensor,
            sample_rate: Optional[int] = None,
            message_threshold: float = 0.5,
    ) -> Tuple[float, torch.Tensor]:
        """
        A convenience function that returns a probability of an audio being watermarked,
        together with its message in n-bits (binary) format. If the audio is not watermarked,
        the message will be random.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio
            message_threshold: threshold used to convert the watermark output (probability
                of each bits being 0 or 1) into the binary n-bit message.
        """
        result, message = self.forward(x)  # b x 2+nbits
        detected = (
                torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1]
        )
        detect_prob = detected.cpu().item()  # type: ignore
        message = torch.gt(message, message_threshold).long()
        return detect_prob, message

    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        """
        Decode the message from the watermark result (batch x nbits x frames)
        Args:
            result: watermark result (batch x nbits x frames)
        Returns:
            The message of size batch x nbits, indicating probability of 1 for each bit
        """
        assert (result.dim() > 2 and result.shape[1] == self.nbits) or (
                self.dim() == 2 and result.shape[0] == self.nbits
        ), f"Expect message of size [,{self.nbits}, frames] (get {result.size()})"
        decoded_message = result.mean(dim=-1)
        return torch.sigmoid(decoded_message)

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect the watermarks from the audio signal
        Args:
            x: Audio signal, size batch x frames
        """
        if self.sampling_rate != 16000:
            x = julius.resample_frac(x, old_sr=self.sampling_rate, new_sr=16000)
        result = self.detector(x)  # [b,2+nbits,16640] torch.Size([1, 18, 16640])
        # hardcode softmax on 2 first units used for detection
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message
