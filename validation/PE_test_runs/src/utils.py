'''
Utility functionality to keep main script clean and organized
'''

import numpy as np
import cupy as cp
from typing import Union

def zero_pad(data: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
    """
    Inputs: data stream of length N
    Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
    """
    xp = cp.get_array_module(data)
    N = len(data)
    pow_2 = xp.ceil(xp.log2(N))
    return xp.pad(data, (0, int((2**pow_2) - N)), 'constant')


def inner_product(
    h: Union[np.ndarray, cp.ndarray],
    d: Union[np.ndarray, cp.ndarray],
    psd: Union[np.ndarray, cp.ndarray],
    df: float,
    dt: float
) -> Union[float, cp.ndarray]:
    '''
    Take usual inner product between two signals
    '''
    xp = cp.get_array_module(h)
    
    assert len(h) == len(d) == len(psd), "Signals and PSD must be of same length"
    
    result = 4 * dt**2 * xp.real(xp.sum((h * d.conj()) / psd*df[1]))
    
    # Return scalar if using cupy
    if xp == cp:
        return result.get() if hasattr(result, 'get') else result
    return result