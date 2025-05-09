"""
plant_cfg.py
Base configuration class for PlantBase.
"""
from torchcontrol.system import SystemCfg, configclass

@configclass
class PlantCfg(SystemCfg):
    """
    Base configuration class for PlantBase.
    """
    def __init__(self):
        super().__init__()
