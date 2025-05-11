"""
input_output_system.py
InputOutputSystem plant: n-order system described by input/output difference equation, using torch.Tensor.
"""
from scipy.signal import tf2ss
from .state_space_system import StateSpaceSystem
from .input_output_system_cfg import InputOutputSystemCfg

class InputOutputSystem(StateSpaceSystem):
    """
    InputOutputSystem plant: n-order system (transfer function form, linear system), implemented via state-space canonical form.
    Args:
        cfg: InputOutputSystemCfg
    """
    def __init__(self, cfg: InputOutputSystemCfg):
        super().__init__(cfg)
        # num/den are handled in cfg and converted to A, B, C, D via tf2ss
        # All state, step, forward, output logic is inherited from StateSpaceSystem
        # No additional attributes or methods are needed unless you want to extend functionality
        pass
    
    def update(self, *args, **kwargs):
        """
        Update the numerator and denominator coefficients of the transfer function.
        Args:
            *args: new numerator and denominator coefficients
            **kwargs: new numerator and denominator coefficients
        """
        for key in ['numerator', 'denominator']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        # Update A, B, C, D matrices based on new numerator and denominator
        A, B, C, D = tf2ss(self.numerator, self.denominator)
        # Call the parent class update method to update the state space matrices
        super().update(A=A, B=B, C=C, D=D)
