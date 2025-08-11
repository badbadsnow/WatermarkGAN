import os
import json
from pathlib import Path
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import julius

import activations
from utils import init_weights, get_padding
from alias_free_activation.torch.act import Activation1d as TorchActivation1d
from env import AttrDict
from seanet.seanet import SEANetEncoder, SEANetDecoder
from audioseal_model import MsgProcessor


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
            self,
            h: AttrDict,
            channels: int,
            kernel_size: int = 3,
            dilation: tuple = (1, 3, 5),
            activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
            self,
            h: AttrDict,
            channels: int,
            kernel_size: int = 3,
            dilation: tuple = (1, 3, 5),
            activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class MessageEncoder(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.embedding = nn.Embedding(2, hidden_dim)
        self.conv2 = nn.Conv1d(32 + hidden_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, message): # [b, 1, nbits] → [b, nbits]
        embedded = self.embedding(message.long())  # [b, nbits, hidden_dim]
        embedded = embedded.permute(0, 2, 1)  # [b, hidden_dim, nbits]
        x = nn.functional.relu(self.conv1(message.unsqueeze(1).float()))  # [b, 32, nbits]
        x = torch.cat([x, embedded], dim=1)  # [b, 32+128, nbits]
        return nn.functional.gelu(self.conv2(x))  # [b, 512, nbits]


class AdaptiveUpsampler(nn.Module):
    def __init__(self, mode='nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, target_length):
        x = nn.functional.interpolate(x, size=target_length, mode=self.mode)
        return x


class ConditionalNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 仿照AdaIN，用消息生成归一化参数
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.gamma_net = nn.Linear(512, channels)  # 消息特征 -> 缩放参数
        self.beta_net = nn.Linear(512, channels)  # 消息特征 -> 偏移参数

    def forward(self, x, msg_feature):
        pooled_msg = nn.functional.adaptive_avg_pool1d(msg_feature, 1)  # [b, 512, 1]
        pooled_msg = pooled_msg.squeeze(-1)  # [b, 512]

        gamma = self.gamma_net(pooled_msg)[:, :, None]  # [b, c, 1]
        beta = self.beta_net(pooled_msg)[:, :, None]  # [b, c, 1]

        normalized = self.norm(x)
        return gamma * normalized + beta


class BigVGAN_AudioSeal(torch.nn.Module):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        # Watermark
        self.sampling_rate = h.get("sampling_rate")
        # self.wm_alpha = nn.Parameter(torch.tensor(1))
        self.wm_alpha = torch.tensor(1)
        self.seanet_configs = h.get("seanet")
        self.decoder_configs = h.get("decoder")
        self.wm_encoders = nn.ModuleList()
        self.msg_processors = nn.ModuleList()
        self.wm_decoders = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            self.wm_encoders.append(
                SEANetEncoder(
                    activation=self.seanet_configs.get("activation"),
                    activation_params=self.seanet_configs.get("activation_params"),
                    causal=self.seanet_configs.get("causal"),
                    channels=ch,
                    compress=self.seanet_configs.get("compress"),
                    dilation_base=self.seanet_configs.get("dilation_base"),
                    dimension=ch // 2,
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
                )
            )
            self.msg_processors.append(
                MsgProcessor(nbits=h.get("nbits", 16), hidden_size=ch // 2)
            )
            self.wm_decoders.append(
                SEANetDecoder(
                    final_activation=self.decoder_configs.get("final_activation"),
                    final_activation_params=self.decoder_configs.get(
                        "final_activation_params"),
                    trim_right_ratio=self.decoder_configs.get("trim_right_ratio"),
                    dimension=ch // 2
                )
            )

    def forward(self, x, message):
        '''
        Args:
            x: torch.Size([b, 100, 32]) mel
            message: torch.Size([b, 1, nbits])
        Returns:
            wav=torch.Size([b, 1, 8192])
        '''

        # Pre-conv
        x = self.conv_pre(x)  # torch.Size([2, 512, 32])

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # torch.Size([2, 256, 256]) torch.Size([2, 128, 2048])  torch.Size([2, 64, 4096])   torch.Size([2, 32, 8192])

            # Watermark
            length = x.size(-1)
            hidden = self.wm_encoders[i](x)
            # torch.Size([2, 128, 1])   torch.Size([2, 16, 26])     torch.Size([2, 32, 13])     torch.Size([2, 16, 26])

            hidden = self.msg_processors[i](hidden, message.to(device=hidden.device))
            # torch.Size([2, 128, 1])   torch.Size([2, 64, 7])      torch.Size([2, 32, 13])     torch.Size([2, 16, 26])

            watermark = self.wm_decoders[i](hidden)[..., :length]
            # torch.Size([2, 1, 256])   torch.Size([2, 1, 2048])    torch.Size([2, 1, 4096])    torch.Size([2, 1, 8192])
            x = x + watermark

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str,
            cache_dir: str,
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",  # Additional argument
            strict: bool = False,  # Additional argument
            use_cuda_kernel: bool = False,
            **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            pass
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            pass

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model


class BigVGAN_AudioSeal_beforeUp512(torch.nn.Module):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        # Watermark
        self.sampling_rate = h.get("sampling_rate")
        # self.wm_alpha = nn.Parameter(torch.tensor(1))
        self.wm_alpha = torch.tensor(1)
        self.seanet_configs = h.get("seanet")
        self.decoder_configs = h.get("decoder")
        self.wm_encoder = SEANetEncoder(
            activation=self.seanet_configs.get("activation"),
            activation_params=self.seanet_configs.get("activation_params"),
            causal=self.seanet_configs.get("causal"),
            channels=self.seanet_configs.get("channels"),
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
        )
        self.msg_processor = MsgProcessor(nbits=h.get("nbits", 16), hidden_size=self.seanet_configs.get("dimension"), )
        self.wm_decoder = SEANetDecoder(
            final_activation=self.decoder_configs.get("final_activation"),
            final_activation_params=self.decoder_configs.get(
                "final_activation_params"),
            trim_right_ratio=self.decoder_configs.get("trim_right_ratio"),
            dimension=self.seanet_configs.get("dimension"),
        )

    def forward(self, x, message):
        '''
        Args:
            x: torch.Size([b, 100, 32]) mel
            message: torch.Size([b, 1, nbits])
        Returns:
            wav=torch.Size([b, 1, 8192])
        '''

        # Pre-conv
        x = self.conv_pre(x)  # torch.Size([2, 512, 32])

        # Watermark
        length = x.size(-1)
        hidden = self.wm_encoder(x)
        hidden = self.msg_processor(hidden, message.to(device=hidden.device))
        watermark = self.wm_decoder(hidden)[..., :length]
        x = x + watermark

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # torch.Size([b, 256, 256]) torch.Size([b, 128, 2048])  torch.Size([b, 64, 4096])   torch.Size([b, 32, 8192])

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)  # torch.Size([b, 32, 8192])
        x = self.conv_post(x)  # torch.Size([b, 32, 8192])

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str,
            cache_dir: str,
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",  # Additional argument
            strict: bool = False,  # Additional argument
            use_cuda_kernel: bool = False,
            **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            pass
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            pass

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model


class BigVGAN_AudioSeal_afterUp32(torch.nn.Module):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        # Watermark
        self.sampling_rate = h.get("sampling_rate")
        # self.wm_alpha = nn.Parameter(torch.tensor(1))
        self.wm_alpha = torch.tensor(1)
        self.seanet_configs = h.get("seanet")
        self.decoder_configs = h.get("decoder")
        self.wm_encoder = SEANetEncoder(
            activation=self.seanet_configs.get("activation"),
            activation_params=self.seanet_configs.get("activation_params"),
            causal=self.seanet_configs.get("causal"),
            channels=self.seanet_configs.get("channels"),
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
        )
        self.msg_processor = MsgProcessor(nbits=h.get("nbits", 16), hidden_size=self.seanet_configs.get("dimension"))
        self.wm_decoder = SEANetDecoder(
            final_activation=self.decoder_configs.get("final_activation"),
            final_activation_params=self.decoder_configs.get(
                "final_activation_params"),
            trim_right_ratio=self.decoder_configs.get("trim_right_ratio"),
            dimension=self.seanet_configs.get("dimension")
        )

    def forward(self, x, message):
        '''
        Args:
            x: torch.Size([b, 100, 32]) mel
            message: torch.Size([b, 1, nbits])
        Returns:
            wav=torch.Size([b, 1, 8192])
        '''

        # Pre-conv
        x = self.conv_pre(x)  # torch.Size([2, 512, 32])

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # torch.Size([2, 256, 256]) torch.Size([2, 128, 2048])  torch.Size([2, 64, 4096])   torch.Size([2, 32, 8192])

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Watermark
        length = x.size(-1)
        hidden = self.wm_encoder(x)
        hidden = self.msg_processor(hidden, message.to(device=hidden.device))
        watermark = self.wm_decoder(hidden)[..., :length]
        x = x + watermark

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str,
            cache_dir: str,
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",  # Additional argument
            strict: bool = False,  # Additional argument
            use_cuda_kernel: bool = False,
            **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            pass
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            pass

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model

class BigVGAN_AudioSeal_withUp(torch.nn.Module):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        # message
        self.wm_alpha = torch.tensor(1)
        self.seanet_configs = h.get("seanet")
        self.decoder_configs = h.get("decoder")
        self.wm_encoder = SEANetEncoder(
            activation=self.seanet_configs.get("activation"),
            activation_params=self.seanet_configs.get("activation_params"),
            causal=self.seanet_configs.get("causal"),
            channels=self.seanet_configs.get("channels"),
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
        )
        self.msg_processor = MsgProcessor(nbits=h.get("nbits", 16), hidden_size=self.seanet_configs.get("dimension"))
        self.wm_decoder = SEANetDecoder(
            final_activation=self.decoder_configs.get("final_activation"),
            final_activation_params=self.decoder_configs.get(
                "final_activation_params"),
            trim_right_ratio=self.decoder_configs.get("trim_right_ratio"),
            dimension=self.seanet_configs.get("dimension")
        )
        self.AdaptiveUpsampler = AdaptiveUpsampler()
        self.ConditionalNorms = nn.ModuleList([
            ConditionalNorm(channels=256 // (2 ** i)) for i in range(self.num_upsamples)
        ])

    def forward(self, x, message):
        '''
        Args:
            x: torch.Size([b, 100, 32]) mel
            message: torch.Size([b, 1, nbits])
        Returns:
            wav=torch.Size([b, 1, 8192])
        '''

        # Pre-conv
        x = self.conv_pre(x)  # torch.Size([2, 512, 32])

        length = x.size(-1)
        hidden = self.wm_encoder(x)
        hidden = self.msg_processor(hidden, message.to(device=hidden.device))
        watermark = self.wm_decoder(hidden)[..., :length]
        msg_encoded = x + watermark

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # torch.Size([2, 256, 256]) torch.Size([2, 128, 2048])  torch.Size([2, 64, 4096])   torch.Size([2, 32, 8192])

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            # message
            target_length = x.shape[-1]
            msg_up = self.AdaptiveUpsampler(msg_encoded, target_length)
            x = self.ConditionalNorms[i](x, msg_up)

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str,
            cache_dir: str,
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",  # Additional argument
            strict: bool = False,  # Additional argument
            use_cuda_kernel: bool = False,
            **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            pass
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            pass

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model

class WmBigVGAN(torch.nn.Module):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        # message
        self.MessageEncoder = MessageEncoder(hidden_dim=128)
        self.AdaptiveUpsampler = AdaptiveUpsampler()
        self.ConditionalNorms = nn.ModuleList([
            ConditionalNorm(channels=256 // (2 ** i)) for i in range(self.num_upsamples)
        ])

    def forward(self, x, message):
        '''
        Args:
            x: torch.Size([b, 100, 32]) mel
            message: torch.Size([b, 1, nbits])
        Returns:
            wav=torch.Size([b, 1, 8192])
        '''

        # Pre-conv
        x = self.conv_pre(x)  # torch.Size([2, 512, 32])

        msg_encoded = self.MessageEncoder(message)  # [b, 512, nbits]

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # torch.Size([2, 256, 256]) torch.Size([2, 128, 2048])  torch.Size([2, 64, 4096])   torch.Size([2, 32, 8192])

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            # message
            target_length = x.shape[-1]
            msg_up = self.AdaptiveUpsampler(msg_encoded, target_length)
            x = self.ConditionalNorms[i](x, msg_up)

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str,
            cache_dir: str,
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",  # Additional argument
            strict: bool = False,  # Additional argument
            use_cuda_kernel: bool = False,
            **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            pass
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            pass

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model

# with open('configs/baseline_base_24khz_100band.json') as f:
#     data = f.read()
# json_config = json.loads(data)
# h = AttrDict(json_config)
# m = BigVGAN_AudioSeal(h)
# x = torch.randn([2, 100, 32])
# message = torch.randint(0, 2, (2, 16))
# y = m(x, message)
# print(x.shape, message.shape, y.shape)
# pass
