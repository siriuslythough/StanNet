"""Custom complex CNN (stanNet) for generic image classification.

- Example architecture showcasing complex layers + attention/stacks.
- Exposes a factory `stanNet_complex(num_classes: int)`.
"""

import torch
from torch import nn

from complexnn import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    ComplexMaxPool2d,
    ComplexAdaptiveAvgPool2d,
    ComplexDropout,
)
from complex_activations import CPReLU  # per-channel slopes help separation

# --------- Utilities ---------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = x.real.new_empty(shape).bernoulli_(keep) / keep
        return x * mask.type_as(x)

def radial_lowpass_mask(h: int, w: int, cutoff: float = 0.25, sharpness: float = 8.0, device=None, dtype=torch.float32):
    # Smooth circular low-pass in frequency domain (0 at corners, 1 near DC)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device, dtype=dtype),
                            torch.linspace(-1, 1, w, device=device, dtype=dtype),
                            indexing='ij')
    rr = torch.sqrt(xx**2 + yy**2)
    # Smooth step: 1 / (1 + exp(k*(r - cutoff)))
    mask = (1.0 / (1.0 + torch.exp(sharpness * (rr - cutoff))))
    return mask  # (H, W) real

# --------- Core novel blocks ---------
class SpectralGate(nn.Module):
    def __init__(self, channels: int, cutoff: float = 0.25):
        super().__init__()
        self.alpha_r = nn.Parameter(torch.zeros(channels))
        self.alpha_i = nn.Parameter(torch.zeros(channels))
        self.beta_r  = nn.Parameter(torch.zeros(channels))
        self.beta_i  = nn.Parameter(torch.zeros(channels))
        self.cutoff = cutoff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Complex-to-complex FFT
        S = torch.fft.fft2(x, norm='ortho')
        # Smooth radial low-pass mask over full (H, W)
        lp = radial_lowpass_mask(H, W, cutoff=self.cutoff, sharpness=8.0,
                                 device=x.device, dtype=x.real.dtype)  # (H, W) real
        lp = lp.unsqueeze(0).unsqueeze(0)                               # (1,1,H,W)
        lp = lp.expand(B, C, H, W).type_as(S)                           # broadcast + complex dtype

        alpha = (self.alpha_r + 1j * self.alpha_i).view(1, C, 1, 1).type_as(S)
        beta  = (self.beta_r  + 1j * self.beta_i ).view(1, C, 1, 1).type_as(S)
        gate = (1.0 + beta) + alpha * lp                                # complex gate

        Sg = S * gate
        xg = torch.fft.ifft2(Sg, norm='ortho')                          # complex inverse FFT
        return xg


class MPCrossGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.m_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        # Input to p_gate will be 2 channels: [cos(phase), sin(phase)]
        self.p_gate = nn.Sequential(
            nn.Conv2d(2, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.abs().real                           # (B,C,H,W) real [attached_file:1]
        p = x.angle().real                         # (B,C,H,W) real [attached_file:1]

        # Phase features as 2 channels, reduced across C to keep 4D for Conv2d
        p2 = torch.stack([torch.cos(p), torch.sin(p)], dim=1)  # (B,2,C,H,W) new dim from stack [web:35]
        p2 = p2.mean(dim=2)                                   # (B,2,H,W) reduce extra axis before Conv2d [web:24]

        gm = self.m_gate(m).type_as(x)            # (B,C,H,W) [attached_file:1]
        gp = self.p_gate(p2).type_as(x)           # (B,C,H,W) [web:24]
        g = 0.5 * (gm + gp)                       # broadcast in complex [attached_file:1]
        return x * g                               # complex gating [attached_file:1]

class ComplexMixerBlock(nn.Module):
    """
    Local complex conv -> BN -> CPReLU
    + SpectralGate (global token mixing)
    + Magnitude–Phase Cross-Gate
    + Residual with DropPath
    """
    def __init__(self, channels: int, drop_path: float = 0.0, spec_cutoff: float = 0.25):
        super().__init__()
        self.conv = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1  = ComplexBatchNorm2d(channels)
        self.act  = CPReLU(num_channels=channels)
        self.spec = SpectralGate(channels, cutoff=spec_cutoff)
        self.bn2  = ComplexBatchNorm2d(channels)
        self.mp   = MPCrossGate(channels, reduction=4)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv(x)))
        y = self.bn2(self.spec(y))
        y = self.mp(y)
        y = self.dp(y)
        return x + y

class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn   = ComplexBatchNorm2d(out_ch)
        self.act  = CPReLU(num_channels=out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class stanNet(nn.Module):
    """
    Complex Mixer with Phase Cross-gating backbone for yoga pose classification.
    - Input: complex (B, 3, H, W)
    - Output: complex logits (B, num_classes, 1, 1)
    """
    def __init__(self, num_classes: int, width: int = 48, depths=(2, 3, 4, 2),
                 drop_path_rate: float = 0.1, spec_cutoff: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            ComplexConv2d(3, width, kernel_size=5, stride=2, padding=2, bias=False),
            ComplexBatchNorm2d(width),
            CPReLU(num_channels=width),
            ComplexMaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, steps=total_blocks).tolist()

        ch = width
        blocks = []
        idx = 0
        stage_channels = [width, width*2, width*4, width*8]
        self.stages = nn.ModuleList()
        for si, (sc, d) in enumerate(zip(stage_channels, depths)):
            if si == 0:
                # First stage: just a 1x1 proj if needed
                if ch != sc:
                    self.stages.append(nn.Sequential(
                        ComplexConv2d(ch, sc, kernel_size=1, bias=False),
                        ComplexBatchNorm2d(sc),
                        CPReLU(num_channels=sc),
                        *[ComplexMixerBlock(sc, drop_path=dpr[idx + j], spec_cutoff=spec_cutoff) for j in range(d)]
                    ))
                else:
                    self.stages.append(nn.Sequential(
                        *[ComplexMixerBlock(sc, drop_path=dpr[idx + j], spec_cutoff=spec_cutoff) for j in range(d)]
                    ))
                idx += d
                ch = sc
            else:
                stage = []
                stage.append(Downsample(ch, sc))
                ch = sc
                for j in range(d):
                    stage.append(ComplexMixerBlock(ch, drop_path=dpr[idx], spec_cutoff=spec_cutoff))
                    idx += 1
                self.stages.append(nn.Sequential(*stage))

        self.pool = ComplexAdaptiveAvgPool2d((1, 1))
        self.drop = ComplexDropout(0.1)
        self.head = ComplexConv2d(ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.head(x)
        return x

def stanNet_complex(num_classes: int, width: int = 48, drop_path_rate: float = 0.1, spec_cutoff: float = 0.25):
    return stanNet(num_classes=num_classes, width=width, depths=(2,3,4,2),
                    drop_path_rate=drop_path_rate, spec_cutoff=spec_cutoff)
