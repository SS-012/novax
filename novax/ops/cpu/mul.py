import numpy as np

"""
CPU fallback elementwise multiplication operation
"""

def mul(a, b):
    return a.data * b.data