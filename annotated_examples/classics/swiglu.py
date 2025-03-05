# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Modifications Copyright 2025 Mekkcyber.  

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE : int = 65536
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def _forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row) # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype)
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)


def swiglu_forward_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _forward_kernel[grid](e, g, h, n_elements, BLOCK_SIZE = 1024,)
    return h


@triton.jit
def _backward_kernel(dY, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    Backward pass for SwiGLU activation function.
    
    Forward pass (for reference):
        f = e * sigmoid(e)  # SiLU/Swish activation
        h = f * g           # Gating mechanism
    
    Backward pass derivation:
        Given dL/dh (dY), we need to compute:
        - dL/de: Gradient with respect to first input
        - dL/dg: Gradient with respect to second input
        
        Using the chain rule:
        dL/dg = dL/dh * dh/dg = dY * f
        dL/de = dL/dh * dh/de = dY * dh/de = dY * g * df/de        
        Where df/de = sigmoid(e) + e * sigmoid(e) * (1 - sigmoid(e))
                    = sigmoid(e) * (1 + e * (1 - sigmoid(e))) (see backprop_math/swiglu.md)
    """
    # Get the block index and calculate offsets for parallel processing
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input tensors
    dY_row = tl.load(dY + offsets, mask = mask, other = 0)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)

    # Compute sigmoid(e) - needed for both forward and backward calculations
    se_row = tl.sigmoid(e_row)  # sigmoid(e)
    
    # Compute f = e * sigmoid(e) (SiLU/Swish activation)
    f_row = se_row * e_row
    f_row = f_row.to(dY_row.dtype)  # Convert back to original dtype
    
    # Compute dL/dg = dY * f
    dg_row = dY_row * f_row
    
    # Compute dL/de = dY * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
    # This is the derivative of SwiGLU with respect to e
    de_row = dY_row.to(tl.float32) * g_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(dY_row.dtype)  # Convert back to original dtype

    # Store computed gradients back to the input buffers
    # Note: We're reusing the input buffers to store the gradients
    tl.store(e + offsets, de_row, mask = mask)  # Store dL/de in e buffer
    tl.store(g + offsets, dg_row, mask = mask)  # Store dL/dg in g buffer


def swiglu_DWf_DW_dfg_kernel(dY, e, g):
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _backward_kernel[grid](dY, e, g, n_elements, BLOCK_SIZE = 1024,)
    return e, g

def test_swiglu_correctness():
    """
    Test the correctness of the SwiGLU implementation by comparing with a reference implementation.
    Tests both forward and backward passes for SwiGLU (SiLU/Swish).
    """
    import torch
    import torch.nn.functional as F

    def swiglu_reference_forward(e, g):
        """Reference implementation of SwiGLU forward pass"""
        return g * (e * F.sigmoid(e))

    forward_kernel = _forward_kernel
    backward_kernel = _backward_kernel
    
    batch_size, seq_len, hidden_dim = 2, 10, 128
    e = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device='cuda')
    g = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device='cuda')

    h = torch.empty_like(e)
    
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    forward_kernel[grid](e, g, h, n_elements, BLOCK_SIZE = 1024)
    
    our_output = h.clone()
    
    ref_output = swiglu_reference_forward(e, g)

    max_diff = torch.max(torch.abs(ref_output - our_output))    
    print(f"Max difference in SwiGLU forward pass: {max_diff.item()}")
    assert max_diff < 1e-5, "SwiGLU forward pass implementation is incorrect!"

    # Test backward pass
    dY = torch.randn_like(h)
    
    # Compute reference gradients
    e.requires_grad_(True)
    g.requires_grad_(True)
    ref_output = swiglu_reference_forward(e, g)
    ref_output.backward(dY)
    ref_de = e.grad.clone()
    ref_dg = g.grad.clone()

    backward_kernel[grid](dY, e, g, n_elements, BLOCK_SIZE = 1024)
    
    max_diff_de = torch.max(torch.abs(ref_de - e))
    print(f"Max difference in SwiGLU backward pass (de): {max_diff_de.item()}")
    assert max_diff_de < 1e-5, "SwiGLU backward pass implementation for de is incorrect!"
    
    max_diff_dg = torch.max(torch.abs(ref_dg - g))
    print(f"Max difference in SwiGLU backward pass (dg): {max_diff_dg.item()}")
    assert max_diff_dg < 1e-5, "SwiGLU backward pass implementation for dg is incorrect!"

    print("All tests passed!")

if __name__ == "__main__":
    test_swiglu_correctness()