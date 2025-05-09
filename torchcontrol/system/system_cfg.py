"""
system_cfg.py
Base configuration class for SystemBase. Uses configclass decorator for style consistency.
"""

def configclass(cls):
    """Decorator to mark config classes (for style consistency)."""
    cls._is_configclass = True
    return cls

@configclass
class SystemCfg:
    """
    Base configuration class for SystemBase.
    """
    def __init__(self):
        pass
