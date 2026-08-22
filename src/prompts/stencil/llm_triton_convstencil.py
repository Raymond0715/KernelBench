import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------- Preprocess (from user) ----------
KY = 7
KX = 7
M8 = 16  # int(BLOCK_M/8) with BLOCK_M=128


def preprocess_conv1_weight(conv_weight):
    """
    Build w_frag (and its flattened view w_frag_flat) according to the polyhedral
    preprocessing schedule:
      w_frag_flat[((p * (KX + 7) + q) * M8) + (yi * 8 + xi)] =
          conv_weight[(p - yi)*KX + (q - xi)]
    Only entries with yi <= p < KY+yi and xi <= q < KX+xi are written; other entries
    are left as zero.
    Args:
      conv_weight: tensor with KY*KX elements (row-major layout expected) or shape (KY,KX).
    Returns:
      w_frag: tensor of shape ((KY+1)*(KX+7), M8)
      w_frag_flat: flattened view of w_frag (1D tensor).
    """
    conv_w = conv_weight.reshape(KY, KX).contiguous()

    rows = (KY + 1) * (KX + 7)
    w_frag = torch.zeros((rows, M8), dtype=conv_w.dtype, device=conv_w.device)

    for yi in range(2):
        for xi in range(8):
            for p in range(yi, KY + yi):
                for q in range(xi, KX + xi):
                    row = p * (KX + 7) + q
                    col = yi * 8 + xi
                    w_frag[row, col] = conv_w[p - yi, q - xi]

    w_frag_flat = w_frag.reshape(-1)
    return w_frag, w_frag_flat


# ---------- Triton kernel (adapted to use program ids) ----------
@triton.jit
def conv_kernel(x_pad_ptr, w_frag_flat_ptr, y_ptr, bias_ptr, H, W, BLOCK_M: tl.constexpr, KY: tl.constexpr = 7, KX: tl.constexpr = 7):
    yt = tl.program_id(0)
    xt = tl.program_id(1)

    pad_w = 3

    # Load input fragment (in_frag) -> shape (16, 128)
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 128)[None, :]

    p = cols // 14
    q = cols % 14

    offsets = (yt * 2 + p) * (W + 2 * pad_w) + (xt * BLOCK_M + rows * 8 + q)
    mask_in = cols < 112

    in_frag = tl.load(x_pad_ptr + offsets, mask=mask_in, other=0.0)

    # Load weight fragment (w_frag) -> shape (128, 16)
    w_rows = tl.arange(0, 128)
    w_cols = tl.arange(0, 16)

    offs_w = w_rows[:, None] * 16 + w_cols[None, :]
    mask_w = (w_rows[:, None] < ((KY + 1) * (KX + 7))) & (w_cols[None, :] < 16)

    w_frag = tl.load(w_frag_flat_ptr + offs_w, mask=mask_w, other=0.0)

    # Compute y fragment
    y_frag = tl.dot(in_frag, w_frag)

    bias_val = tl.load(bias_ptr)
    y_frag = y_frag + bias_val

    # Store y_frag into output y using linear offsets and mask
    a = tl.arange(0, 16)
    b = tl.arange(0, 16)

    row_offsets = yt * 2 + (b // 8)[None, :]
    row_offsets = row_offsets + a[:, None] * 0

    col_offsets = xt * BLOCK_M + a[:, None] * 8 + (b % 8)[None, :]

    mask_out = (row_offsets < H) & (col_offsets < W)

    # Compute linear offsets (since tl.store doesn't accept 2D offsets keyword)
    linear_offsets = row_offsets * W + col_offsets

    tl.store(y_ptr + linear_offsets, y_frag, mask=mask_out)


# ---------- Launcher and Model ----------
BLOCK_M = 128  # compile-time constant from IR


def triton_conv_launcher(x, w_frag_flat_preproc, bias, *, BLOCK_M=BLOCK_M):
    """
    Launcher that prepares padded input, moves preprocessed weights to device,
    constructs grid from the polyhedral IR and launches conv_kernel.
    """
    assert x.is_cuda, "Input must be on CUDA."

    orig_dim = x.dim()
    added_batch = False
    added_channel = False
    if orig_dim == 2:
        x = x[None, None, :, :]
        added_batch = True
        added_channel = True
    elif orig_dim == 3:
        x = x[None, :, :, :]
        added_batch = True
    elif orig_dim == 4:
        pass
    else:
        raise ValueError("Unsupported input dimensions")

    B, C, H, W = x.shape
    pad_h = 3
    pad_w = 3

    assert H % 2 == 0, f"H must be divisible by 2 (got H={H})"
    assert W % BLOCK_M == 0, f"W must be divisible by BLOCK_M={BLOCK_M} (got W={W})"

    w_frag_flat_dev = w_frag_flat_preproc.to(device=x.device, dtype=x.dtype).contiguous()
    bias_dev = bias.detach().to(device=x.device, dtype=x.dtype).reshape(-1).contiguous()

    out = torch.zeros_like(x)

    grid_h = H // 2
    grid_w = W // BLOCK_M
    grid = (grid_h, grid_w)

    for b in range(B):
        for c in range(C):
            x_bc = x[b, c]
            x_pad = F.pad(x_bc, pad=(pad_w, pad_w, pad_h, pad_h), mode='constant', value=0).contiguous()
            x_pad_flat = x_pad.reshape(-1)

            y_view = out[b, c]
            y_buf = y_view if y_view.is_contiguous() else y_view.contiguous()

            conv_kernel[grid](
                x_pad_flat,
                w_frag_flat_dev,
                y_buf,
                bias_dev,
                H, W,
                BLOCK_M=BLOCK_M, KY=KY, KX=KX
            )

            if y_buf is not y_view:
                out[b, c].copy_(y_buf)

    if added_batch and added_channel:
        return out[0, 0]
    if added_batch:
        return out[0]
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
        conv_w = self.conv1.weight.detach()[0, 0].cpu().contiguous()
        _, w_frag_flat = preprocess_conv1_weight(conv_w)
        self.register_buffer("_w_frag_flat_preproc", w_frag_flat, persistent=False)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors for this Triton implementation.")

        bias = self.conv1.bias.detach().reshape(1)
        y = triton_conv_launcher(x, self._w_frag_flat_preproc, bias, BLOCK_M=BLOCK_M)
        return y


# ---------- Simple sanity test (only run when executed as a script) ----------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available - skipping runtime smoke test.")
    else:
        model = ModelNew().cuda()
        B = 1
        C = 1
        H = 128
        W = 128
        x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            y = model(x)
        print("Output shape:", y.shape)
