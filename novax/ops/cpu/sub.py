import numpy as np

"""
CPU fallback elementwise subtraction operation
"""

def sub(a, b):
    return a.data - b.data