import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dints_block import (
    ActiConvNormBlock,
    FactorizedIncreaseBlock,
    FactorizedReduceBlock,
    P3DActiConvNormBlock,
)
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import optional_import

# solving shortest path problem
csr_matrix, _ = optional_import("scipy.sparse", name="csr_matrix")
dijkstra, _ = optional_import("scipy.sparse.csgraph", name="dijkstra")

@torch.jit.interface
class CellInterface(torch.nn.Module):
    """interface for torchscriptable Cell"""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass


@torch.jit.interface
class StemInterface(torch.nn.Module):
    """interface for torchscriptable Stem"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass


class StemTS(StemInterface):
    """wrapper for torchscriptable Stem"""

    def __init__(self, *mod):
        super().__init__()
        self.mod = torch.nn.Sequential(*mod)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x)  # type: ignore


def _dfs(node, paths):
    """use depth first search to find all path activation combination"""
    if node == paths:
        return [[0], [1]]
    child = _dfs(node + 1, paths)
    return [[0] + _ for _ in child] + [[1] + _ for _ in child]


class _IdentityWithRAMCost(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ram_cost = 0


class _CloseWithRAMCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.ram_cost = 0

    def forward(self, x):
        return torch.tensor(0.0, requires_grad=False).to(x)


class _ActiConvNormBlockWithRAMCost(ActiConvNormBlock):
    """The class wraps monai layers with ram estimation. The ram_cost = total_ram/output_size is estimated.
    Here is the estimation:
     feature_size = output_size/out_channel
     total_ram = ram_cost * output_size
     total_ram = in_channel * feature_size (activation map) +
                 in_channel * feature_size (convolution map) +
                 out_channel * feature_size (normalization)
               = (2*in_channel + out_channel) * output_size/out_channel
     ram_cost = total_ram/output_size = 2 * in_channel/out_channel + 1
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, kernel_size, padding, spatial_dims, act_name, norm_name)
        self.ram_cost = 1 + in_channel / out_channel * 2


class _P3DActiConvNormBlockWithRAMCost(P3DActiConvNormBlock):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        p3dmode: int = 0,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, kernel_size, padding, p3dmode, act_name, norm_name)
        # 1 in_channel (activation) + 1 in_channel (convolution) +
        # 1 out_channel (convolution) + 1 out_channel (normalization)
        self.ram_cost = 2 + 2 * in_channel / out_channel


class _FactorizedIncreaseBlockWithRAMCost(FactorizedIncreaseBlock):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, spatial_dims, act_name, norm_name)
        # s0 is upsampled 2x from s1, representing feature sizes at two resolutions.
        # 2 * in_channel * s0 (upsample + activation) + 2 * out_channel * s0 (conv + normalization)
        # s0 = output_size/out_channel
        self.ram_cost = 2 * in_channel / out_channel + 2


class _FactorizedReduceBlockWithRAMCost(FactorizedReduceBlock):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, spatial_dims, act_name, norm_name)
        # s0 is upsampled 2x from s1, representing feature sizes at two resolutions.
        # in_channel * s0 (activation) + 3 * out_channel * s1 (convolution, concatenation, normalization)
        # s0 = s1 * 2^(spatial_dims) = output_size / out_channel * 2^(spatial_dims)
        self.ram_cost = in_channel / out_channel * 2**self._spatial_dims + 3


class MixedOp(nn.Module):
    """
    The weighted averaging of cell operations.
    Args:
        c: number of output channels.
        ops: a dictionary of operations. See also: ``Cell.OPS2D`` or ``Cell.OPS3D``.
        arch_code_c: binary cell operation code. It represents the operation results added to the output.
    """

    def __init__(self, c: int, ops: dict, arch_code_c=None):
        super().__init__()
        if arch_code_c is None:
            arch_code_c = np.ones(len(ops))
        self.ops = nn.ModuleList()
        for arch_c, op_name in zip(arch_code_c, ops):
            self.ops.append(_CloseWithRAMCost() if arch_c == 0 else ops[op_name](c))

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            x: input tensor.
            weight: learnable architecture weights for cell operations. arch_code_c are derived from it.
        Return:
            out: weighted average of the operation results.
        """
        out = 0.0
        weight = weight.to(x)
        for idx, _op in enumerate(self.ops):
            out = out + _op(x) * weight[idx]
        return out


class Cell(CellInterface):
    """
    The basic class for cell operation search, which contains a preprocessing operation and a mixed cell operation.
    Each cell is defined on a `path` in the topology search space.
    Args:
        c_prev: number of input channels
        c: number of output channels
        rate: resolution change rate. It represents the preprocessing operation before the mixed cell operation.
            ``-1`` for 2x downsample, ``1`` for 2x upsample, ``0`` for no change of resolution.
        arch_code_c: cell operation code
    """

    DIRECTIONS = 3
    # Possible output paths for `Cell`.
    #
    #       - UpSample
    #      /
    # +--+/
    # |  |--- Identity or AlignChannels
    # +--+\
    #      \
    #       - Downsample

    # Define 2D operation set, parameterized by the number of channels
    OPS2D = {
        "skip_connect": lambda _c: _IdentityWithRAMCost(),
        "conv_3x3": lambda c: _ActiConvNormBlockWithRAMCost(c, c, 3, padding=1, spatial_dims=2),
    }

    # Define 3D operation set, parameterized by the number of channels
    OPS3D = {
        "skip_connect": lambda _c: _IdentityWithRAMCost(),
        "conv_3x3x3": lambda c: _ActiConvNormBlockWithRAMCost(c, c, 3, padding=1, spatial_dims=3),
        "conv_3x3x1": lambda c: _P3DActiConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=0),
        "conv_3x1x3": lambda c: _P3DActiConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=1),
        "conv_1x3x3": lambda c: _P3DActiConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=2),
    }

    # Define connection operation set, parameterized by the number of channels
    ConnOPS = {
        "up": _FactorizedIncreaseBlockWithRAMCost,
        "down": _FactorizedReduceBlockWithRAMCost,
        "identity": _IdentityWithRAMCost,
        "align_channels": _ActiConvNormBlockWithRAMCost,
    }

    def __init__(
        self,
        c_prev: int,
        c: int,
        rate: int,
        arch_code_c=None,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__()
        self._spatial_dims = spatial_dims
        self._act_name = act_name
        self._norm_name = norm_name

        if rate == -1:  # downsample
            self.preprocess = self.ConnOPS["down"](
                c_prev, c, spatial_dims=self._spatial_dims, act_name=self._act_name, norm_name=self._norm_name
            )
        elif rate == 1:  # upsample
            self.preprocess = self.ConnOPS["up"](
                c_prev, c, spatial_dims=self._spatial_dims, act_name=self._act_name, norm_name=self._norm_name
            )
        else:
            if c_prev == c:
                self.preprocess = self.ConnOPS["identity"]()
            else:
                self.preprocess = self.ConnOPS["align_channels"](
                    c_prev, c, 1, 0, spatial_dims=self._spatial_dims, act_name=self._act_name, norm_name=self._norm_name
                )

        # Define 2D operation set, parameterized by the number of channels
        self.OPS2D = {
            "skip_connect": lambda _c: _IdentityWithRAMCost(),
            "conv_3x3": lambda c: _ActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, spatial_dims=2, act_name=self._act_name, norm_name=self._norm_name
            ),
        }

        # Define 3D operation set, parameterized by the number of channels
        self.OPS3D = {
            "skip_connect": lambda _c: _IdentityWithRAMCost(),
            "conv_3x3x3": lambda c: _ActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, spatial_dims=3, act_name=self._act_name, norm_name=self._norm_name
            ),
            "conv_3x3x1": lambda c: _P3DActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, p3dmode=0, act_name=self._act_name, norm_name=self._norm_name
            ),
            "conv_3x1x3": lambda c: _P3DActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, p3dmode=1, act_name=self._act_name, norm_name=self._norm_name
            ),
            "conv_1x3x3": lambda c: _P3DActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, p3dmode=2, act_name=self._act_name, norm_name=self._norm_name
            ),
        }

        self.OPS = {}
        if self._spatial_dims == 2:
            self.OPS = self.OPS2D
        elif self._spatial_dims == 3:
            self.OPS = self.OPS3D
        else:
            raise NotImplementedError(f"Spatial dimensions {self._spatial_dims} is not supported.")

        self.op = MixedOp(c, self.OPS, arch_code_c)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
            weight: weights for different operations.
        """
        x = self.preprocess(x)
        x = self.op(x, weight)
        return x

class TopologyConstruction(nn.Module):
    """
    The base class for `TopologyInstance` and `TopologySearch`.

    Args:
        arch_code: `[arch_code_a, arch_code_c]`, numpy arrays. The architecture codes defining the model.
            For example, for a ``num_depths=4, num_blocks=12`` search space:

            - `arch_code_a` is a 12x10 (10 paths) binary matrix representing if a path is activated.
            - `arch_code_c` is a 12x10x5 (5 operations) binary matrix representing if a cell operation is used.
            - `arch_code` in ``__init__()`` is used for creating the network and remove unused network blocks. If None,

            all paths and cells operations will be used, and must be in the searching stage (is_search=True).
        channel_mul: adjust intermediate channel number, default is 1.
        cell: operation of each node.
        num_blocks: number of blocks (depth in the horizontal direction) of the DiNTS search space.
        num_depths: number of image resolutions of the DiNTS search space: 1, 1/2, 1/4 ... in each dimension.
        use_downsample: use downsample in the stem. If False, the search space will be in resolution [1, 1/2, 1/4, 1/8],
            if True, the search space will be in resolution [1/2, 1/4, 1/8, 1/16].
        device: `'cpu'`, `'cuda'`, or device ID.


    Predefined variables:
        `filter_nums`: default to 32. Double the number of channels after downsample.
        topology related variables:

            - `arch_code2in`: path activation to its incoming node index (resolution). For depth = 4,
              arch_code2in = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]. The first path outputs from node 0 (top resolution),
              the second path outputs from node 1 (second resolution in the search space),
              the third path outputs from node 0, etc.
            - `arch_code2ops`: path activation to operations of upsample 1, keep 0, downsample -1. For depth = 4,
              arch_code2ops = [0, 1, -1, 0, 1, -1, 0, 1, -1, 0]. The first path does not change
              resolution, the second path perform upsample, the third perform downsample, etc.
            - `arch_code2out`: path activation to its output node index.
              For depth = 4, arch_code2out = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
              the first and second paths connects to node 0 (top resolution), the 3,4,5 paths connects to node 1, etc.
    """

    def __init__(
        self,
        arch_code: Optional[list] = None,
        channel_mul: float = 1.0,
        cell=Cell,
        num_blocks: int = 6,
        num_depths: int = 3,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        use_downsample: bool = True,
        device: str = "cpu",
    ):

        super().__init__()

        self.filter_nums = [int(n_feat * channel_mul) for n_feat in (32, 64, 128, 256, 512)]
        self.num_blocks = num_blocks
        self.num_depths = num_depths
        self._spatial_dims = spatial_dims
        self._act_name = act_name
        self._norm_name = norm_name
        self.use_downsample = use_downsample
        self.device = device
        self.num_cell_ops = 0
        if self._spatial_dims == 2:
            self.num_cell_ops = len(cell.OPS2D)
        elif self._spatial_dims == 3:
            self.num_cell_ops = len(cell.OPS3D)

        # Calculate predefined parameters for topology search and decoding
        arch_code2in, arch_code2out = [], []
        for i in range(Cell.DIRECTIONS * self.num_depths - 2):
            arch_code2in.append((i + 1) // Cell.DIRECTIONS - 1 + (i + 1) % Cell.DIRECTIONS)
        arch_code2ops = ([-1, 0, 1] * self.num_depths)[1:-1]
        for m in range(self.num_depths):
            arch_code2out.extend([m, m, m])
        arch_code2out = arch_code2out[1:-1]
        self.arch_code2in = arch_code2in
        self.arch_code2ops = arch_code2ops
        self.arch_code2out = arch_code2out

        # define NAS search space
        if arch_code is None:
            arch_code_a = torch.ones((self.num_blocks, len(self.arch_code2out))).to(self.device)
            arch_code_c = torch.ones((self.num_blocks, len(self.arch_code2out), self.num_cell_ops)).to(self.device)
        else:
            arch_code_a = torch.from_numpy(arch_code[0]).to(self.device)
            arch_code_c = F.one_hot(torch.from_numpy(arch_code[1]).to(torch.int64), self.num_cell_ops).to(self.device)

        self.arch_code_a = arch_code_a
        self.arch_code_c = arch_code_c
        # define cell operation on each path
        self.cell_tree = nn.ModuleDict()
        for blk_idx in range(self.num_blocks):
            for res_idx in range(len(self.arch_code2out)):
                if self.arch_code_a[blk_idx, res_idx] == 1:
                    self.cell_tree[str((blk_idx, res_idx))] = cell(
                        self.filter_nums[self.arch_code2in[res_idx] + int(use_downsample)],
                        self.filter_nums[self.arch_code2out[res_idx] + int(use_downsample)],
                        self.arch_code2ops[res_idx],
                        self.arch_code_c[blk_idx, res_idx],
                        self._spatial_dims,
                        self._act_name,
                        self._norm_name,
                    )

    def forward(self, x):
        """This function to be implemented by the architecture instances or search spaces."""
        pass

class TopologyInstance(TopologyConstruction):
    """
    Instance of the final searched architecture. Only used in re-training/inference stage.
    """

    def __init__(
        self,
        arch_code=None,
        channel_mul: float = 1.0,
        cell=Cell,
        num_blocks: int = 6,
        num_depths: int = 3,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        use_downsample: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize DiNTS topology search space of neural architectures.
        """
        if arch_code is None:
            warnings.warn("arch_code not provided when not searching.")

        super().__init__(
            arch_code=arch_code,
            channel_mul=channel_mul,
            cell=cell,
            num_blocks=num_blocks,
            num_depths=num_depths,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            use_downsample=use_downsample,
            device=device,
        )


    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: input tensor.
        """
        # generate path activation probability
        inputs, outputs = x, [torch.tensor(0.0).to(x[0])] * self.num_depths
        for blk_idx in range(self.num_blocks):
            outputs = [torch.tensor(0.0).to(x[0])] * self.num_depths
            for res_idx, activation in enumerate(self.arch_code_a[blk_idx].data):
                if activation:
                    mod: CellInterface = self.cell_tree[str((blk_idx, res_idx))]
                    _out = mod.forward(
                        x=inputs[self.arch_code2in[res_idx]], weight=torch.ones_like(self.arch_code_c[blk_idx, res_idx])
                    )
                    outputs[self.arch_code2out[res_idx]] = outputs[self.arch_code2out[res_idx]] + _out
            inputs = outputs

        return inputs

class DiNTS(nn.Module):
    """
    Reimplementation of DiNTS based on
    "DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation
    <https://arxiv.org/abs/2103.15954>".

    The model contains a pre-defined multi-resolution stem block (defined in this class) and a
    DiNTS space (defined in :py:class:`monai.networks.nets.TopologyInstance` and
    :py:class:`monai.networks.nets.TopologySearch`).

    The stem block is for: 1) input downsample and 2) output upsample to original size.
    The model downsamples the input image by 2 (if ``use_downsample=True``).
    The downsampled image is downsampled by [1, 2, 4, 8] times (``num_depths=4``) and used as input to the
    DiNTS search space (``TopologySearch``) or the DiNTS instance (``TopologyInstance``).

        - ``TopologyInstance`` is the final searched model. The initialization requires the searched architecture codes.
        - ``TopologySearch`` is a multi-path topology and cell operation search space.
          The architecture codes will be initialized as one.
        - ``TopologyConstruction`` is the parent class which constructs the instance and search space.

    To meet the requirements of the structure, the input size for each spatial dimension should be:
    divisible by 2 ** (num_depths + 1).

    Args:
        dints_space: DiNTS search space. The value should be instance of `TopologyInstance` or `TopologySearch`.
        in_channels: number of input image channels.
        num_classes: number of output segmentation classes.
        act_name: activation name, default to 'RELU'.
        norm_name: normalization used in convolution blocks. Default to `InstanceNorm`.
        spatial_dims: spatial 2D or 3D inputs.
        use_downsample: use downsample in the stem.
            If ``False``, the search space will be in resolution [1, 1/2, 1/4, 1/8],
            if ``True``, the search space will be in resolution [1/2, 1/4, 1/8, 1/16].
        node_a: node activation numpy matrix. Its shape is `(num_depths, num_blocks + 1)`.
            +1 for multi-resolution inputs.
            In model searching stage, ``node_a`` can be None. In deployment stage, ``node_a`` cannot be None.
    """

    def __init__(
        self,
        dints_space,
        in_channels: int,
        num_classes: int,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        spatial_dims: int = 3,
        use_downsample: bool = True,
        node_a=None,
    ):
        super().__init__()

        self.dints_space = dints_space
        self.filter_nums = dints_space.filter_nums
        self.num_blocks = dints_space.num_blocks
        self.num_depths = dints_space.num_depths
        if spatial_dims not in (2, 3):
            raise NotImplementedError(f"Spatial dimensions {spatial_dims} is not supported.")
        self._spatial_dims = spatial_dims
        if node_a is None:
            self.node_a = torch.ones((self.num_blocks + 1, self.num_depths))
        else:
            self.node_a = node_a

        # define stem operations for every block
        conv_type = Conv[Conv.CONV, spatial_dims]
        self.stem_down = nn.ModuleDict()
        self.stem_up = nn.ModuleDict()
        mode = "trilinear" if self._spatial_dims == 3 else "bilinear"
        for res_idx in range(self.num_depths):
            # define downsample stems before DiNTS search
            if use_downsample:
                self.stem_down[str(res_idx)] = StemTS(
                    nn.Upsample(scale_factor=1 / (2**res_idx), mode=mode, align_corners=True),
                    conv_type(
                        in_channels=in_channels,
                        out_channels=self.filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx]),
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=self.filter_nums[res_idx],
                        out_channels=self.filter_nums[res_idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx + 1]),
                )
                self.stem_up[str(res_idx)] = StemTS(
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=self.filter_nums[res_idx + 1],
                        out_channels=self.filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx]),
                    nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
                )

            else:
                self.stem_down[str(res_idx)] = StemTS(
                    nn.Upsample(scale_factor=1 / (2**res_idx), mode=mode, align_corners=True),
                    conv_type(
                        in_channels=in_channels,
                        out_channels=self.filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx]),
                )
                self.stem_up[str(res_idx)] = StemTS(
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=self.filter_nums[res_idx],
                        out_channels=self.filter_nums[max(res_idx - 1, 0)],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(
                        name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[max(res_idx - 1, 0)]
                    ),
                    nn.Upsample(scale_factor=2 ** (res_idx != 0), mode=mode, align_corners=True),
                )

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]

    def forward(self, x: torch.Tensor):
        """
        Prediction based on dynamic arch_code.

        Args:
            x: input tensor.
        """
        inputs = []
        for d in range(self.num_depths):
            # allow multi-resolution input
            _mod_w: StemInterface = self.stem_down[str(d)]
            x_out = _mod_w.forward(x)
            if self.node_a[0][d]:
                inputs.append(x_out)
            else:
                inputs.append(torch.zeros_like(x_out))

        outputs = self.dints_space(inputs)

        blk_idx = self.num_blocks - 1
        start = False
        _temp: torch.Tensor = torch.empty(0)
        for res_idx in range(self.num_depths - 1, -1, -1):
            _mod_up: StemInterface = self.stem_up[str(res_idx)]
            if start:
                _temp = _mod_up.forward(outputs[res_idx] + _temp)
            elif self.node_a[blk_idx + 1][res_idx]:
                start = True
                _temp = _mod_up.forward(outputs[res_idx])

        return outputs[-1], _temp

if __name__ == "__main__":
    ckpt = torch.load('./arch_code_cvpr.pth')
    node_a = ckpt["node_a"]
    arch_code_a = ckpt["arch_code_a"]
    arch_code_c = ckpt["arch_code_c"]

    dints_space = TopologyInstance(
            channel_mul=1.0,
            num_blocks=12,
            num_depths=4,
            use_downsample=True,
            arch_code=[arch_code_a, arch_code_c]
        )

    net = DiNTS(
            dints_space=dints_space,
            in_channels=1,
            num_classes=3,
            use_downsample=True,
            node_a=node_a,
        )
    input_tensor = torch.zeros(1, 1, 96, 96, 96)
    net(input_tensor)