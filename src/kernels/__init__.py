"""Kernels for graphs."""


from .base import Kernel

from .count import CountingKernel
from .node_histogram import NodeHistogramKernel
from .edge_histogram import EdgeHistogramKernel
from .order_walk import OrderWalkKernel
from .geometric_walk import GeometricWalkKernel
from .shortest_path import ShortestPathKernel
from .sum import SumKernel


__all__ = [
    "Kernel",

    "CountingKernel",
    "NodeHistogramKernel",
    "EdgeHistogramKernel",
    "OrderWalkKernel",
    "GeometricWalkKernel",
    "ShortestPathKernel",
    "SumKernel"
]
