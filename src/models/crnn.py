"""全卷积 CRNN：per-frame 多标签异常检测。

输入  : [B, 1, n_mels + n_td, T] —— log-mel（+ 可选 n_td 行逐帧时域统计）
主干  : N 个 ConvBlock 只处理 log-mel 行（仅沿 mel 池化，保留时间维）
时域支路: n_td 行逐帧统计 -> Linear+ReLU，与卷积特征拼接后送入 GRU
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
        n_td: int = 0,
        td_hidden: int = 32,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernel) == len(pool), "channels/kernel/pool 长度须一致"
        self.n_mels = n_mels
        self.n_classes = n_classes
        self.gru_hidden = gru_hidden
        self.n_layers = n_layers
        self.n_td = n_td

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
        if n_td:
            self.td_proj = nn.Sequential(nn.Linear(n_td, td_hidden), nn.ReLU(inplace=True))
            gru_in += td_hidden
        else:
            self.td_proj = None

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
        from src.features.mel import N_TD

        m = cfg["model"]
        spec = m[m["name"]]  # small | base 配置块
        n_td = N_TD if cfg.get("features", {}).get("time_features", False) else 0
        return cls(
            n_mels=m.get("n_mels", 128),
            n_classes=m.get("n_classes", 7),
            channels=spec["channels"],
            kernel=spec["kernel"],
            pool=spec["pool"],
            gru_hidden=spec["gru_hidden"],
            dropout=spec.get("dropout", 0.2),
            n_td=n_td,
            td_hidden=spec.get("td_hidden", 32),
        )

    @property
    def n_features(self) -> int:
        return self.n_mels + self.n_td

    def _split(self, x: torch.Tensor):
        if x.dim() == 3:  # [B,F,T] -> 加通道
            x = x.unsqueeze(1)
        if x.shape[2] != self.n_features:
            raise ValueError(f"Expected {self.n_features} feature rows "
                             f"({self.n_mels} mel + {self.n_td} time-domain), got {x.shape[2]}")
        return x[:, :, : self.n_mels], (x[:, 0, self.n_mels :] if self.n_td else None)

    def _sequence(self, conv: torch.Tensor, td: torch.Tensor | None) -> torch.Tensor:
        B, C, M, T = conv.shape
        seq = conv.reshape(B, C * M, T).permute(0, 2, 1)  # [B, T, C*mel']
        if td is not None:
            seq = torch.cat([seq, self.td_proj(td.permute(0, 2, 1))], dim=-1)
        return seq.contiguous()

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        """x: [B,1,n_mels,T] -> (logits [B,n_classes,T], new_hidden)。

        hidden 为 None 时自动初始化。流式增量推理时传入上一步返回的 hidden。
        """
        mel, td = self._split(x)
        x = self._sequence(self.conv(mel), td)  # [B, T, C*mel' (+td_hidden)]
        B = x.shape[0]

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
        mel, td = self._split(x)
        context, hidden = state if state is not None else (None, None)
        n_new = mel.shape[-1]
        if n_new == 0:
            raise ValueError("Expected at least one frame")
        joined = mel if context is None else torch.cat([context, mel], dim=-1)
        keep = sum(block._causal_time for block in self.conv)
        conv = self.conv(joined)[..., -n_new:]
        seq = self._sequence(conv, td)
        out, hidden = self.gru(seq, hidden)
        logits = self.head(out).permute(0, 2, 1).contiguous()
        context = joined[..., -keep:].detach() if keep else None
        return logits, (context, hidden.detach())

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Mapping) -> CRNN:
    return CRNN.from_cfg(cfg)
