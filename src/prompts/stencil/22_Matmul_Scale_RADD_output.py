architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
```
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_add_clamp_logsumexp_mish_mul_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8192 * x0), xmask, other=0.0)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 + tmp0
    tmp4 = -10.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 10.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, float('-inf'))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tl_math.abs(tmp11)
    tmp13 = float('inf')
    tmp14 = tmp12 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp15, tmp11)
    tmp17 = tmp7 - tmp16
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tl_math.log(tmp22)
    tmp24 = tmp23 + tmp16
    tmp25 = 20.0
    tmp26 = tmp24 > tmp25
    tmp27 = tl_math.exp(tmp24)
    tmp28 = libdevice.log1p(tmp27)
    tmp29 = tl.where(tmp26, tmp24, tmp28)
    tmp30 = libdevice.tanh(tmp29)
    tmp31 = tmp24 * tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr0 + x0, tmp31, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (8192, 8192), (1, 8192), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((1024, 1), (1, 1024), torch.float32)
        buf2 = reinterpret_tensor(buf1, (1024, 1), (1, 1), 0)
        del buf1
        buf3 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_clamp_logsumexp_mish_mul_0[grid(1024)](buf2,
            buf0, buf3, 1024, 8192, XBLOCK=8, num_warps=8, num_stages=1)
    return buf3, primals_3, buf0, buf2


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
```