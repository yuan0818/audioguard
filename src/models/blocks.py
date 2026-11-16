"""卷积块：Conv2d + BN + ReLU + 沿 mel 轴池化。

时间轴不被全局池化，保留 per-frame 输出以支持流式。
池化仅作用在 mel（频率）轴，逐块压缩频率维、扩大通道。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """二维卷积块。

    结构: Conv2d(kernel) -> BatchNorm2d -> ReLU -> (可选) AvgPool2d 沿 mel 池化。

    Args:
        in_ch / out_ch: 通道数。
        kernel: 时-频核尺寸 (kt, kf)。
        pool: 沿 mel 池化尺寸 (0 表示不池化)。
        dropout: dropout 概率。
        causal_time: 是否在时间轴上因果填充（左填充），用于流式。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple[int, int] = (3, 3),
        pool: int = 2,
        dropout: float = 0.0,
        causal_time: bool = False,
    ) -> None:
        super().__init__()
        kt, kf = kernel
        # 时间因果填充（仅左填充），使输出帧只依赖过去帧；否则对称填充会有少量未来偷看。
        if causal_time:
            self._causal_time = kt - 1
            self._time_pad = (kt - 1, 0)  # (left, right)
        else:
            self._causal_time = 0
            self._time_pad = (kt // 2, kt // 2)
        pf = kf // 2  # 频率轴对称填充
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kt, kf), padding=(pf, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.AvgPool2d(kernel_size=(pool, 1)) if pool and pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, n_mels, T]
        if self._causal_time:
            left, right = self._time_pad
            # F.pad 末两维: (W_left, W_right, H_left, H_right); W=time
            x = F.pad(x, (left, right, 0, 0))
        x = self.conv(x)               # mel 轴由 padding=(pf,0) 对称填充
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)               # 沿 mel 池化
        return x
