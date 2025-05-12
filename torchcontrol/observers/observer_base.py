from __future__ import annotations

import abc
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observer_cfg import ObserverCfg


class ObserverBase(nn.Module, metaclass=abc.ABCMeta):
    cfg: ObserverCfg

    def __init__(self, cfg: ObserverCfg):
        self.cfg = cfg