"""全卷积 CRNN：per-frame 多标签异常检测。

输入  : [B, 1, n_mels, T]   —— log-mel 谱图（加通道维）
主干  : N 个 ConvBlock（仅沿 mel 池化，保留时间维）
时序头: 单向 GRU（因果，可流式带状态）
输出头: 逐帧线性 -> [B, n_classes, T] logits（多标签，每帧独立 sigmoid）

同一权重同时用于离线整段与流式增量推理：流式时携带 GRU 隐状态、按 hop 推理新帧。
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from src.models.blocks import ConvBlock


class CRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 7,
        channels: list[int] = (32, 64, 96, 128),
        kernel: list[int] = (3, 3, 3, 3),
        pool: list[int] = (2, 2, 2, 2),
        gru_hidden: int = 128,
        dropout: float = 0.2,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernel) == len(pool), "channels/kernel/pool 长度须一致"
        self.n_mels = n_mels
        self.n_classes = n_classes
        self.gru_hidden = gru_hidden
        self.n_layers = n_layers

        blocks: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            blocks.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    kernel=(kernel[i], kernel[i]),
                    pool=pool[i],
                    dropout=dropout if i > 0 else 0.0,
                    causal_time=True,  # 时间因果填充，便于流式
                )
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        # 池化后 mel 维度
        mel_reduced = n_mels
        for p in pool:
            if p and p > 1:
                mel_reduced = mel_reduced // p
        self.mel_reduced = max(mel_reduced, 1)
        self.last_ch = channels[-1]
        gru_in = self.last_ch * self.mel_reduced

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=gru_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(gru_hidden, n_classes)

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> "CRNN":
        m = cfg["model"]
        spec = m[m["name"]]  # small | base 配置块
        return cls(
            n_mels=m.get("n_mels", 128),
            n_classes=m.get("n_classes", 7),
            channels=spec["channels"],
            kernel=spec["kernel"],
            pool=spec["pool"],
            gru_hidden=spec["gru_hidden"],
            dropout=spec.get("dropout", 0.2),
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        """x: [B,1,n_mels,T] -> (logits [B,n_classes,T], new_hidden)。

        hidden 为 None 时自动初始化。流式增量推理时传入上一步返回的 hidden。
        """
        if x.dim() == 3:  # [B,n_mels,T] -> 加通道
            x = x.unsqueeze(1)
        x = self.conv(x)  # [B, C, mel', T]
        B, C, M, T = x.shape
        x = x.reshape(B, C * M, T)        # [B, C*mel', T]
        x = x.permute(0, 2, 1).contiguous()  # [B, T, C*mel']

        if hidden is None:
            hidden = x.new_zeros(self.n_layers, B, self.gru_hidden)
        out, hidden = self.gru(x, hidden)  # [B,T,H], hidden
        logits = self.head(out)            # [B,T,n_classes]
        logits = logits.permute(0, 2, 1).contiguous()  # [B,n_classes,T]
        return logits, hidden

    def forward_stream(self, x, state=None):
        """Eval-only incremental forward; state belongs to the caller, not the model.

        Preserve the convolution receptive field and GRU hidden state. This
        produces the same frames as a full causal forward in eval mode.
        """
        if self.training:
            raise RuntimeError("Streaming requires eval mode")
        if x.dim() == 3:
            x = x.unsqueeze(1)
        context, hidden = state if state is not None else (None, None)
        n_new = x.shape[-1]
        if n_new == 0:
            raise ValueError("Expected at least one frame")
        joined = x if context is None else torch.cat([context, x], dim=-1)
        keep = sum(block._causal_time for block in self.conv)
        conv = self.conv(joined)[..., -n_new:]
        B, C, M, T = conv.shape
        seq = conv.reshape(B, C * M, T).permute(0, 2, 1).contiguous()
        out, hidden = self.gru(seq, hidden)
        logits = self.head(out).permute(0, 2, 1).contiguous()
        context = joined[..., -keep:].detach() if keep else None
        return logits, (context, hidden.detach())

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Mapping) -> CRNN:
    return CRNN.from_cfg(cfg)
