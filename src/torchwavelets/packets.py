"""Wavelet packets (full binary decomposition tree) for 1D and 2D.

A wavelet packet decomposes both the approximation AND the detail bands at
every level, producing a full binary (1D) or quaternary (2D) tree. The API
mirrors ``pywt.WaveletPacket`` / ``pywt.WaveletPacket2D``: nodes are accessed
by path (``"aad"``, ``"av"``, ...), the tree is built with ``decompose()``,
reconstructed with ``reconstruct()``, and coefficients match PyWavelets
bit-for-bit (verified by the parity tests).

All node data are torch tensors of shape ``(N, C, L)`` (1D) or
``(N, C, H, W)`` (2D); the precision follows the input dtype.
"""

from __future__ import annotations

import torch

from .core.filters import filter_bank
from .dwt._validation import dwt_max_level
from .dwt.transform1d import dwt1d, idwt1d
from .dwt.transform2d import dwt2d, idwt2d

__all__ = ["WaveletPacket", "WaveletPacket2D"]


def get_graycode_order(level: int, x: str = "a", y: str = "d") -> list[str]:
    """Sequency (frequency) ordering of the paths at ``level``."""
    order = [x, y]
    for _ in range(level - 1):
        order = [x + p for p in order] + [y + p for p in order[::-1]]
    return order


class BaseNode:
    """A node of a wavelet packet tree."""

    PART_LEN = 1
    PARTS: tuple[str, ...] = ()

    def __init__(self, parent, data, node_name):
        self.parent = parent
        if parent is not None:
            self.wavelet = parent.wavelet
            self.mode = parent.mode
            self.level = parent.level + 1
            self._maxlevel = parent.maxlevel
            self.path = parent.path + node_name
        else:
            self.wavelet = None
            self.mode = None
            self._maxlevel = None
            self.path = ""
            self.level = 0
        self.axes = getattr(parent, "axes", None) if parent is not None else None
        self._data = data
        for part in self.PARTS:
            setattr(self, part, None)

    # -- data ------------------------------------------------------------
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def is_empty(self) -> bool:
        return self._data is None

    @property
    def maxlevel(self) -> int:
        return self._maxlevel

    @property
    def is_leaf(self) -> bool:
        return not any(getattr(self, part) is not None for part in self.PARTS)

    @property
    def has_any_subnode(self) -> bool:
        return any(getattr(self, part) is not None for part in self.PARTS)

    # -- tree structure ---------------------------------------------------
    def _get_node(self, part):
        return getattr(self, part)

    def _set_node(self, part, node):
        setattr(self, part, node)

    def _validate_node_name(self, part):
        if part not in self.PARTS:
            raise ValueError(f"Subnode name must be in {self.PARTS}, not {part!r}.")

    def _create_subnode(self, part, data=None, overwrite=True):
        self._validate_node_name(part)
        if not overwrite and self._get_node(part) is not None:
            return self._get_node(part)
        node = type(self)(self, data, part)
        self._set_node(part, node)
        return node

    def get_subnode(self, part, decompose=True):
        """Return the direct subnode ``part``, decomposing if needed."""
        node = self._get_node(part)
        if node is None:
            if decompose:
                self._decompose()
                node = self._get_node(part)
                if node is None:
                    raise ValueError(f"Invalid subnode name: {part!r}")
            else:
                raise ValueError(f"Subnode {part!r} does not exist.")
        return node

    def decompose(self):
        """Decompose this node one level (children are (re)created)."""
        if self.level >= self.maxlevel:
            raise ValueError("Maximum decomposition level reached.")
        return self._decompose()

    def _reconstruct(self) -> torch.Tensor:
        """Rebuild this node's data from its subnodes (implemented by the
        1-D and 2-D trees)."""
        raise NotImplementedError

    def _order_freq(self, nodes, level):
        """Sequency ordering of the nodes at ``level`` (implemented by the
        1-D and 2-D trees)."""
        raise NotImplementedError

    def reconstruct(self, update: bool = False) -> torch.Tensor:
        """Reconstruct data from subnodes (IDWT), or return leaf data.

        With ``update=True`` the reconstructed data replaces the node data.
        """
        if not self.has_any_subnode:
            return self.data
        data = self._reconstruct()
        if update:
            self.data = data
        return data

    def get_path(self, path: str):
        """Return the node at ``path``, decomposing as needed."""
        if not path:
            return self
        return self.get_subnode(path[: self.PART_LEN], True).get_path(path[self.PART_LEN :])

    def walk(self, func, decompose=True):
        """Depth-first walk; ``func(node)`` returns False to stop descent."""
        if func(self) is False:
            return
        if not self.has_any_subnode and decompose and self.level < self.maxlevel:
            self._decompose()
        for part in self.PARTS:
            node = self._get_node(part)
            if node is not None:
                node.walk(func, decompose)

    def get_leaf_nodes(self, decompose=False):
        """Return all leaf nodes (with ``decompose=True``, fill the tree)."""
        result = []

        def collect(node):
            if node.is_leaf:
                result.append(node)
                return False
            return True

        self.walk(collect, decompose=decompose)
        return result

    def get_level(self, level: int, order: str = "natural", decompose=True):
        """Return all nodes at ``level`` (``order`` = 'natural' or 'freq')."""
        if order not in ("natural", "freq"):
            raise ValueError(f"Invalid order: {order!r}")
        if level > self.maxlevel:
            raise ValueError(
                "The level cannot be greater than the maximum "
                f"decomposition level value ({self.maxlevel})"
            )
        result = []

        def collect(node):
            if node.level == level:
                result.append(node)
                return False
            return True

        self.walk(collect, decompose=decompose)
        if order == "natural":
            return result
        return self._order_freq(result, level)

    # -- python accessors -------------------------------------------------
    def __getitem__(self, path):
        if isinstance(path, tuple):
            path = "".join(path)
        if not isinstance(path, str):
            raise TypeError(
                f"Invalid path parameter type - expected string or tuple of "
                f"strings but got {type(path)}."
            )
        if self.maxlevel is not None and len(path) > self.maxlevel * self.PART_LEN:
            raise IndexError("Path length is out of range.")
        if not path:
            return self
        return self.get_subnode(path[: self.PART_LEN], True)[path[self.PART_LEN :]]

    def __setitem__(self, path, data):
        if isinstance(path, tuple):
            path = "".join(path)
        if not isinstance(path, str):
            raise TypeError(f"Invalid path parameter type - expected string but got {type(path)}.")
        if self.maxlevel is not None and len(self.path) + len(path) > self.maxlevel * self.PART_LEN:
            raise IndexError("Path length out of range.")
        if path:
            subnode = self._get_node(path[: self.PART_LEN])
            if subnode is None:
                self._create_subnode(path[: self.PART_LEN], None)
                subnode = self._get_node(path[: self.PART_LEN])
            subnode[path[self.PART_LEN :]] = data
        else:
            if isinstance(data, BaseNode):
                self.data = data.data
            else:
                self.data = torch.as_tensor(data)

    def __repr__(self):
        return f"{type(self).__name__}(path={self.path!r}, level={self.level})"


class WaveletPacket(BaseNode):
    """1-D wavelet packet decomposition tree.

    Args:
        data (torch.Tensor): ``(N, C, L)`` input (a 1-D tensor is promoted to
            ``(1, 1, L)``).
        wavelet (str or pywt.Wavelet or tuple): as in :func:`torchwavelets.dwt1d`.
        mode (str): as in :func:`torchwavelets.dwt1d`.
        maxlevel (int): maximum decomposition depth (default: pywt's rule).
        axis (int): the signal axis (default ``-1``, any of the three dims).

    Children are named ``'a'`` (approximation) and ``'d'`` (detail).
    """

    PARTS = ("a", "d")
    CHILD: type[BaseNode] | None = None  # set after the Node class is defined

    def __init__(self, data, wavelet, mode="symmetric", maxlevel=None, axis=-1):
        super().__init__(None, data, "")
        self.wavelet = wavelet
        self.mode = mode
        self.axes = axis
        if data is not None:
            if data.ndim == 1:
                data = data.unsqueeze(0).unsqueeze(0)
            if data.ndim != 3:
                raise ValueError(f"WaveletPacket expects (N, C, L) data; got {tuple(data.shape)}")
            self._data = data
            self.data_size = data.shape
            fl = filter_bank(wavelet).length
            if maxlevel is None:
                maxlevel = dwt_max_level(data.shape[self._axis()], fl)
        self._maxlevel = maxlevel

    def _create_subnode(self, part, data=None, overwrite=True):
        self._validate_node_name(part)
        if not overwrite and self._get_node(part) is not None:
            return self._get_node(part)
        node = self.CHILD(self, data, part)
        self._set_node(part, node)
        return node

    def _axis(self) -> int:
        """Normalise the (already validated) transform axis to ``-1``-style."""
        axis = self.axes
        if axis not in (0, 1, 2, -1, -2, -3):
            raise ValueError(f"invalid axis for a (N, C, L) tensor: {axis!r}")
        return axis if axis >= 0 else axis + 3

    def _to_last(self, x: torch.Tensor) -> torch.Tensor:
        ax = self._axis()
        return x if ax == 2 else x.movedim(ax, -1)

    def _from_last(self, x: torch.Tensor) -> torch.Tensor:
        ax = self._axis()
        return x if ax == 2 else x.movedim(-1, ax)

    def _decompose(self):
        yl, (yh,) = dwt1d(self._to_last(self.data), self.wavelet, self.mode, level=1)
        self._data_shape = tuple(self.data.shape)
        self._create_subnode("a", self._from_last(yl))
        self._create_subnode("d", self._from_last(yh))
        return self.a, self.d

    def _reconstruct(self):
        rec = idwt1d(
            (self._to_last(self.a.reconstruct()), [self._to_last(self.d.reconstruct())]),
            self.wavelet,
            self.mode,
        )
        rec = self._from_last(rec)
        if self._data_shape is not None and tuple(rec.shape) != self._data_shape:
            rec = rec[tuple(slice(sz) for sz in self._data_shape)]
        return rec

    def _order_freq(self, nodes, level):
        by_path = {n.path: n for n in nodes}
        return [by_path[p] for p in get_graycode_order(level, "a", "d") if p in by_path]


class Node(WaveletPacket):
    """A 1-D wavelet packet tree node (internal; created by the tree)."""

    def __init__(self, parent, data, node_name):
        BaseNode.__init__(self, parent, data, node_name)


WaveletPacket.CHILD = Node


class WaveletPacket2D(BaseNode):
    """2-D wavelet packet decomposition tree.

    Args:
        data (torch.Tensor): ``(N, C, H, W)`` input (a 2-D tensor is promoted
            to ``(1, 1, H, W)``).
        wavelet (str or pywt.Wavelet or tuple): as in :func:`torchwavelets.dwt2d`.
        mode (str): as in :func:`torchwavelets.dwt2d`.
        maxlevel (int): maximum decomposition depth (default: pywt's rule).
        axes (tuple[int, int]): the transformed axes (``(-2, -1)`` only).

    Children are named ``'a'`` (LL), ``'h'`` (LH), ``'v'`` (HL), ``'d'`` (HH),
    following PyWavelets' convention (``'h'``/``'v'`` match ``cH``/``cV``).
    """

    PARTS = ("a", "h", "v", "d")
    CHILD: type[BaseNode] | None = None

    def __init__(self, data, wavelet, mode="smooth", maxlevel=None, axes=(-2, -1)):
        super().__init__(None, data, "")
        self.wavelet = wavelet
        self.mode = mode
        self.axes = tuple(axes)
        self._axes()  # validate at construction (children inherit)
        if data is not None:
            if data.ndim == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            if data.ndim != 4:
                raise ValueError(
                    f"WaveletPacket2D expects (N, C, H, W) data; got {tuple(data.shape)}"
                )
            self._data = data
            self.data_size = data.shape
            fl = filter_bank(wavelet).length
            if maxlevel is None:
                maxlevel = dwt_max_level(min(data.shape[-2:]), fl)
        self._maxlevel = maxlevel

    def _create_subnode(self, part, data=None, overwrite=True):
        self._validate_node_name(part)
        if not overwrite and self._get_node(part) is not None:
            return self._get_node(part)
        node = self.CHILD(self, data, part)
        self._set_node(part, node)
        return node

    def _axes(self) -> tuple[int, int]:
        """Normalise and validate the transformed axes."""
        a = tuple(axes % 4 for axes in self.axes)
        if a != (2, 3):
            raise ValueError(
                "WaveletPacket2D supports axes=(-2, -1) only (the last two "
                f"dims of (N, C, H, W)); got {tuple(self.axes)}"
            )
        return a

    def _decompose(self):
        self._axes()  # validate the transform axes
        yl, (h,) = dwt2d(self.data, self.wavelet, self.mode, level=1)
        self._data_shape = tuple(self.data.shape)
        cH, cV, cD = torch.unbind(h, dim=2)
        self._create_subnode("a", yl)
        self._create_subnode("h", cH)
        self._create_subnode("v", cV)
        self._create_subnode("d", cD)
        return self.a, self.h, self.v, self.d

    def _reconstruct(self):
        rec = idwt2d(
            (
                self.a.reconstruct(),
                [
                    torch.stack(
                        (
                            self.h.reconstruct(),
                            self.v.reconstruct(),
                            self.d.reconstruct(),
                        ),
                        dim=2,
                    )
                ],
            ),
            self.wavelet,
            self.mode,
        )
        if self._data_shape is not None and tuple(rec.shape) != self._data_shape:
            rec = rec[tuple(slice(sz) for sz in self._data_shape)]
        return rec

    def _order_freq(self, nodes, level):
        # Observed PyWavelets convention: 'a'->(l,l), 'h'->(h,l), 'v'->(l,h),
        # 'd'->(h,h) (first char = row band, second = column band).
        def expand(path):
            table = {"a": "ll", "h": "hl", "v": "lh", "d": "hh"}
            return (
                "".join(table[c][0] for c in path),
                "".join(table[c][1] for c in path),
            )

        grid = {}
        for node in nodes:
            row, col = expand(node.path)
            grid.setdefault(row, {})[col] = node
        gray = get_graycode_order(level, "l", "h")
        rows = [grid[p] for p in gray if p in grid]
        return [[row[p] for p in gray if p in row] for row in rows]


class Node2D(WaveletPacket2D):
    """A 2-D wavelet packet tree node (internal; created by the tree)."""

    def __init__(self, parent, data, node_name):
        BaseNode.__init__(self, parent, data, node_name)


WaveletPacket2D.CHILD = Node2D
