import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_convolution_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1000
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x1 = xindex % 128
    x2 = xindex // 128
    tmp4 = tl.load(in_ptr0 + 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, YBLOCK])
    tmp19 = tl.load(in_ptr2 + 0)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, YBLOCK])
    tmp0 = y0
    tmp1 = tl.full([1, 1], -3, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.load(in_ptr1 + (-3 + x3 + 128 * y0), tmp2 & xmask & ymask,
        eviction_policy='evict_last', other=0.0)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp6 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr1 + (-128 + x3 + 128 * y0), tmp15 & xmask &
        ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp5 * tmp16
    tmp18 = tmp6 + tmp17
    tmp21 = tmp20 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tmp0 >= tmp13
    tmp26 = tl.full([1, 1], 2, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr1 + (-255 + x3 + 128 * y0), tmp28 & xmask &
        ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp5 * tmp29
    tmp31 = tmp6 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp28, tmp31, tmp32)
    tmp34 = tmp0 >= tmp26
    tmp35 = tl.full([1, 1], 3, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr1 + (-384 + x3 + 128 * y0), tmp37 & xmask &
        ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp5 * tmp38
    tmp40 = tmp6 + tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tmp0 >= tmp35
    tl.full([1, 1], 4, tl.int64)
    tmp46 = tl.load(in_ptr1 + (-512 + x3 + 128 * y0), tmp43 & xmask &
        ymask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp5 * tmp46
    tmp48 = tmp6 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp43, tmp48, tmp49)
    tmp51 = tl.where(tmp37, tmp42, tmp50)
    tmp52 = tl.where(tmp28, tmp33, tmp51)
    tmp53 = tl.where(tmp15, tmp24, tmp52)
    tmp54 = tl.where(tmp2, tmp11, tmp53)
    tl.store(out_ptr0 + (y0 + 1000 * x3), tmp54, xmask & ymask)


@triton.jit
def triton_poi_fused_add_convolution_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 128 % 1000
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr1 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_2, (1, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_3, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1000, 128, 128), (1638400, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_0[grid(1000, 128)](primals_3,
            primals_1, primals_2, buf0, 1000, 128, XBLOCK=128, YBLOCK=8,
            num_warps=4, num_stages=1)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((1, 1000, 128, 128), (1638400, 16384, 128,
            1), torch.float32)
        triton_poi_fused_add_convolution_1[grid(1638400)](buf0, primals_1,
            buf1, 1638400, XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
        del primals_1
    return buf1,


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, input_0):
        primals_2 = self.conv1.weight
        primals_3 = self.conv1.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
