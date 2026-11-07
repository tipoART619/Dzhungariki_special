"""Нейросеть синтеза ДОЭ: физически-информированная U-Net.

Схема (см. README):
  4 изображения t_j
    -> амплитуда t_j * exp(i * phi_obj_j)       (phi_obj_j — обучаемая «фаза объекта»)
    -> обратное распространение Френеля в плоскость ДОЭ для каждого z_j: V_j
    -> вход U-Net: Re/Im V_j (8 каналов) + Re/Im суммы S = sum_j V_j (2 канала)
    -> U-Net предсказывает поправку (dRe, dIm) к S
    -> фаза ДОЭ = atan2(Im S + dIm, Re S + dRe)  in [-pi, pi]

Сеть не должна «выучить» преобразование Фурье — эту часть делает известная
физика (обратный пропагатор). Сеть учится исправлять суперпозиционную голограмму
так, чтобы подавить перекрёстные помехи и спекл (то, что обычно делают сотни
итераций GS). Последний слой инициализирован нулями, поэтому в начале обучения
сеть выдаёт суперпозиционную голограмму, а не шум.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .optics_torch import FresnelPropagator


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU(),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * min(2 ** i, 8) for i in range(depth + 1)]
        self.inc = ConvBlock(cin, chs[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(depth):
            self.down.append(nn.Conv2d(chs[i], chs[i], 3, stride=2, padding=1))
            self.enc.append(ConvBlock(chs[i], chs[i + 1]))
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.dec.append(ConvBlock(chs[i + 1] + chs[i], chs[i]))
        self.out = nn.Conv2d(chs[0], cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        h = skips[0]
        for down, enc in zip(self.down, self.enc):
            h = enc(down(h))
            skips.append(h)
        skips.pop()
        for dec in self.dec:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, s], 1))
        return self.out(h)


class DOENet(nn.Module):
    def __init__(self, propagator: FresnelPropagator, base: int = 32, depth: int = 4, seed: int = 0):
        super().__init__()
        self.prop = propagator
        d, n = propagator.num_planes, propagator.n
        g = torch.Generator().manual_seed(seed)
        self.obj_phase = nn.Parameter((torch.rand(d, n, n, generator=g) * 2 - 1) * math.pi)
        self.unet = UNet(2 * d + 2, 2, base, depth)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """targets: (B, D, n, n) в [0, 1] -> фаза ДОЭ (B, n, n) в [-pi, pi]."""
        u0 = torch.polar(targets, self.obj_phase.expand_as(targets))
        v = self.prop.backward_field(u0)                       # (B, D, n, n)
        s = v.sum(1)                                           # (B, n, n)
        scale = s.abs().pow(2).mean((-2, -1)).sqrt().clamp_min(1e-8)[:, None, None]
        v = v / scale[:, None]
        s = s / scale
        x = torch.cat([v.real, v.imag, s.real[:, None], s.imag[:, None]], 1)
        d = self.unet(x)
        re = s.real + d[:, 0]
        im = s.imag + d[:, 1]
        return torch.atan2(im, re + 1e-12)
