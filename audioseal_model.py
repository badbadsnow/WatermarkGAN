import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import julius
import torch
import torch.nn as nn

logger = logging.getLogger("Audioseal")

COMPATIBLE_WARNING = """
AudioSeal is designed to work at a sample rate 16khz.
Implicit sampling rate usage is deprecated and will be removed in future version.
To remove this warning please add this argument to the function call:
sample_rate = your_sample_rate
"""

class MsgProcessor(nn.Module):
    """Processes secret messages for watermark embedding"""

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor requires nbits > 0"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_embed = nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        indices = 2 * torch.arange(msg.shape[-1], device=msg.device)
        indices = indices.repeat(msg.shape[0], 1) + msg.long()
        msg_aux = self.msg_embed(indices).sum(dim=1)
        return hidden + msg_aux.unsqueeze(-1).expand_as(hidden)


class AudioSealWM(nn.Module):
    """Generator model for AudioSeal watermarks"""

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            msg_processor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor
        self._message: Optional[torch.Tensor] = None

    @property
    def message(self) -> Optional[torch.Tensor]:
        return self._message

    @message.setter
    def message(self, msg: torch.Tensor) -> None:
        self._message = msg

    def get_watermark(
            self,
            x: torch.Tensor,
            sample_rate: Optional[int] = None,
            message: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        length = x.size(-1)
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000

        if sample_rate != 16000:
            x = julius.resample_frac(x, sample_rate, 16000)

        hidden = self.encoder(x)

        if self.msg_processor and message is None:
            message = self.message.to(x.device) if self.message else torch.randint(
                0, 2, (x.size(0), self.msg_processor.nbits), device=x.device
            )

        if self.msg_processor and message is not None:
            hidden = self.msg_processor(hidden, message.to(x.device))

        wm = self.decoder(hidden)

        if sample_rate != 16000:
            wm = julius.resample_frac(wm, 16000, sample_rate)

        return wm[..., :length]

    def forward(
            self,
            x: torch.Tensor,
            sample_rate: Optional[int] = None,
            message: Optional[torch.Tensor] = None,
            alpha: float = 1.0,
    ) -> torch.Tensor:
        wm = self.get_watermark(x, sample_rate, message)
        return x + alpha * wm


class AudioSealDetector(nn.Module):
    """Detector model for AudioSeal watermarks"""

    def __init__(self, encoder: nn.Module, nbits: int = 0):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.nbits = nbits

    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(result.mean(dim=-1))

    def detect_watermark(
            self,
            x: torch.Tensor,
            sample_rate: Optional[int] = None,
            threshold: float = 0.5,
    ) -> Tuple[float, torch.Tensor]:
        detect_out, msg_out = self.forward(x, sample_rate)
        detect_prob = (detect_out[:, 1, :] > 0.5).float().mean().item()
        msg_binary = (msg_out > threshold).int()
        return detect_prob, msg_binary

    def forward(
            self,
            x: torch.Tensor,
            sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000

        if sample_rate != 16000:
            x = julius.resample_frac(x, sample_rate, 16000)

        features = self.encoder(x)
        output = self.classifier(features)

        # Detection scores (first 2 channels)
        detect_scores = torch.softmax(output[:, :2], dim=1)

        # Message scores (remaining channels)
        msg_scores = self.decode_message(output[:, 2:])

        return detect_scores, msg_scores


# Configuration classes from doc1
@dataclass
class SEANetConfig:
    channels: int
    dimension: int
    n_filters: int
    n_residual_layers: int
    ratios: List[int]
    activation: str
    activation_params: Dict[str, float] = field(default_factory=dict)
    norm: str = "none"
    norm_params: Dict[str, Any] = field(default_factory=dict)
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    causal: bool = False
    pad_mode: str = "reflect"
    true_skip: bool = True
    compress: int = 2
    lstm: int = 0
    disable_norm_outer_blocks: int = 0


@dataclass
class DecoderConfig:
    final_activation: Optional[str] = None
    final_activation_params: Optional[dict] = None
    trim_right_ratio: float = 1.0


@dataclass
class DetectorConfig:
    output_dim: int = 32


@dataclass
class AudioSealWMConfig:
    nbits: int
    seanet: SEANetConfig
    decoder: DecoderConfig


@dataclass
class AudioSealDetectorConfig:
    nbits: int
    seanet: SEANetConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)