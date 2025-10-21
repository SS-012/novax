import numpy as np

"""
CPU fallback elementwise addition operation
"""

def add(a, b):
    return a.data + b.data

