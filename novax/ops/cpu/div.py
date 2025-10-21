import numpy as np

"""
CPU fallback elementwise division operation
"""

def div(a, b):
    return a.data / b.data