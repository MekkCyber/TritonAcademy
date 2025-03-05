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

from triton.language.extra import libdevice
triton_tanh = libdevice.tanh
triton_cast = tl.cast
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
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    Forward pass for the GeGLU (Gated Gaussian Error Linear Unit) activation function.
    
    This kernel computes:
    1. The GELU activation: f = 0.5 * e * (1 + erf(e/sqrt(2)))
    2. The GeGLU output: h = f * g
    
    Parameters:
    - e: gate values (first half of the projection)
    - g: up values (second half of the projection)
    - h: output tensor to store the result
    - n_elements: total number of elements in the tensors
    - BLOCK_SIZE: size of each CUDA block for parallelization
    """
    # Get the current block index in the grid
    block_idx = tl.program_id(0)
    
    # Calculate memory offsets for this block
    # Each block thread processes BLOCK_SIZE elements from the input
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where n_elements is not divisible by BLOCK_SIZE
    mask = offsets < n_elements

    # Load input values from global memory
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)

    # Compute GELU activation using the exact formula:
    # f(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    # where erf is the error function
    # rsqrt(2.0) computes 1/sqrt(2)
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    
    # Convert the result back to the same dtype as g_row
    f_row = f_row.to(g_row.dtype)
    
    # Compute the final GeGLU output by multiplying the GELU activation with g element-wise
    h_row = f_row * g_row

    # Store the result back to global memory
    tl.store(h + offsets, h_row, mask = mask)


def geglu_exact_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype = gate.dtype, device = "cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _exact_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE = 1024,)
    return out


@triton.jit
def _exact_backward_kernel(dY, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    Backward pass for the GeGLU (Gated Gaussian Error Linear Unit) activation function.
    
    In the forward pass:
    - f = 1/2 * e * (1 + erf(1/sqrt(2) * e))  # The GELU function
    - h = f * g                               # The GeGLU output
    
    Where:
    - e: gate values (first half of the projection)
    - g: up values (second half of the projection)
    - h: output of GeGLU
    
    In the backward pass, we need to compute:
    - de: gradient with respect to e (gate values)
    - dg: gradient with respect to g (up values)
    
    For de, we need the derivative of f with respect to e:
    df/de = 1/2 * (1 + erf(1/sqrt(2) * x)) + 1/sqrt(2*pi) * x * exp(-1/2 * x^2) (see backprop_math/geglu.md)
    
    Parameters:
    - dY: gradient flowing from the next layer (dL/dh)
    - e: gate values from forward pass
    - g: up values from forward pass
    - n_elements: total number of elements in the tensors
    - BLOCK_SIZE: size of each CUDA block for parallelization
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the gradients and values
    dY_row = tl.load(dY + offsets, mask = mask, other = 0)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)

    # Compute the partial GELU activation: 1/2 * (1 + erf(1/sqrt(2) * e))
    # This is reused in both the forward computation and the derivative
    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    
    # Complete the GELU computation: f = f_partial_row * e_row
    f_row = f_partial_row * e_row
    f_row = f_row.to(dY_row.dtype)

    # Compute gradient for g: dg = dY * f
    # By chain rule: dL/dg = dL/dh * dh/dg = dY * f (as specified above h is the output of the GeGLU)
    dg_row = dY_row * f_row
    
    # Compute gradient for e using the derivative of GELU
    # df/de = f_partial_row + (1/sqrt(2*pi)) * e * exp(-e²/2)
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)
    
    # Apply chain rule: dL/de = dL/dh * dh/de = dY * (g * df/de)
    de_row = g_row.to(tl.float32) * df_de 
    de_row = de_row.to(dY_row.dtype) * dY_row

    # Store the computed gradients back to memory
    tl.store(e  + offsets, de_row, mask = mask)
    tl.store(g  + offsets, dg_row, mask = mask)


def geglu_exact_backward_kernel(DW, e, g):
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _exact_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE = 1024,)
    return e, g


@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    Computes the forward pass of the approximate GELU activation function for GeGLU.
    
    GeGLU (Gated GELU Linear Unit) combines a gating mechanism with GELU activation:
    - Input is split into two parts: gate (e) and up (g)
    - GELU is applied to the gate values
    - The result is multiplied by the up values
    
    This kernel implements the approximate version of GELU which is faster but slightly less accurate.
    
    Formula:
    f(e) = 0.5 * e * (1 + tanh(sqrt(2/π) * e * (1 + 0.044715 * e²)))
    h = f(e) * g
    
    Where:
    - e: gate values
    - g: up values
    - h: output
    - f(e): approximate GELU activation
    """
    # Get the current block index and compute offsets for each thread
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Constant for the GELU approximation: sqrt(2/π)
    s = 0.7978845608028654  # Precomputed value of sqrt(2/π)
    
    # Load gate and up values from memory
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)
    
    # Compute the approximate GELU activation:
    # f(e) = 0.5 * e * (1 + tanh(sqrt(2/π) * e * (1 + 0.044715 * e²)))
    # 
    # This is a faster approximation of the exact GELU:
    # f(e) = 0.5 * e * (1 + erf(e/sqrt(2)))
    #
    # The approximation uses tanh instead of erf and adds a cubic term
    # to better match the shape of the exact GELU function
    inner_term = s * e_row * (1.0 + 0.044715 * e_row * e_row)
    f_row = 0.5 * e_row * (triton_tanh(inner_term) + 1.0)
    
    # Convert back to the original data type
    f_row = f_row.to(g_row.dtype)
    
    # Compute the final output: h = f(e) * g
    h_row = f_row * g_row

    # Store the result back to memory
    tl.store(h + offsets, h_row, mask = mask)


def geglu_approx_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype = gate.dtype, device = "cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _approx_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE = 1024,)
    return out

@triton.jit
def _approx_backward_kernel(dY, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    Backward pass for the approximate GELU activation function.
    
    Forward pass:
    f(e) = 0.5 * e * (1 + tanh(sqrt(2/π) * e * (1 + 0.044715 * e²)))
    h = f(e) * g
    
    Where:
    - e: gate values
    - g: up values
    - dY: gradient from upstream layers
    - h: output
    
    Backward pass derivatives:
    1. df/de = 0.5 * (1 + tanh(inner))(1 + e * (2 - (1 + tanh(inner))) * d(inner)/de)
       where inner = sqrt(2/π) * e * (1 + 0.044715 * e²)
    2. d(inner)/de = sqrt(2/π) * (1 + 0.044715 * e² * 3) = (a + 3b * e²) where a = sqrt(2/π) and b = 0.044715*sqrt(2/π)
    3. de = dY * g * df/de
    4. dg = dY * f(e)
    
    """
    # Get block index and compute offsets for parallel processing
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values from memory
    dY_row = tl.load(dY + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # Constants for the GELU approximation
    a = 0.7978845608028654  # Precomputed value of sqrt(2/π)
    b = 0.044715 * a

    # Full inner term: sqrt(2/π) * e * (1 + 0.044715 * e²)
    inner = e_row * (a + b * e_row * e_row)
    
    # Compute tanh of the inner term
    tanh_inner = triton_tanh(inner)
    
    # Compute (1 + tanh(inner_term))
    v = 1.0 + tanh_inner
    
    # compute f(e) = 0.5 * e * (1 + tanh(inner_term)) based on the forward pass formula in the backprop_math/geglu.md
    f_row = 0.5 * e_row * v

    # compute df/de based on the fomula in the backprop_math/geglu.md
    df_de = 0.5 * v * (1.0 + e_row * (2.0 - v) * (a + 3.0 * b * e_row * e_row))
    
    # Compute gradients for backpropagation:
    # dg = dY * f(e)
    dg_row = dY_row * f_row
    
    # Compute gradients for backpropagation:
    # de = dY * g * df/de
    de_row = g_row * df_de
    de_row = de_row.to(dY_row.dtype) * dY_row

    # Store results and gradients back to memory
    tl.store(e + offsets, de_row, mask=mask)   # Store gradient for gate values
    tl.store(g + offsets, dg_row, mask=mask)     # Store gradient for up values


def geglu_approx_backward_kernel(dY, e, g):
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _approx_backward_kernel[grid](dY, e, g, n_elements, BLOCK_SIZE = 1024,)
    return e, g


def test_geglu_correctness(use_approx=False):
    """
    Test the correctness of the GEGLU implementation by comparing with a reference implementation.
    Tests both forward and backward passes for GEGLU (Gated GELU).
    
    Args:
        use_approx (bool): If True, use the approximate GEGLU implementation.
                          If False, use the exact GEGLU implementation.
    """
    import torch
    import torch.nn.functional as F
    
    # Define reference implementations for GEGLU (Gated GELU)
    def geglu_reference_forward(x):
        """Reference implementation of GEGLU forward pass"""
        x_chunks = torch.chunk(x, 2, dim=-1)
        gate, value = x_chunks[0], x_chunks[1]
        return value * F.gelu(gate)
    
    # Select the appropriate kernels based on the use_approx flag
    forward_kernel = _approx_forward_kernel if use_approx else _exact_forward_kernel
    backward_kernel = _approx_backward_kernel if use_approx else _exact_backward_kernel
    
    implementation_type = "approximate" if use_approx else "exact"
    print(f"Testing {implementation_type} GEGLU implementation...")
    
    def test_forward():
        """Test the forward pass of GEGLU"""
        print(f"Testing {implementation_type} GEGLU forward pass...")
        
        batch_size, seq_len, hidden_dim = 2, 10, 128
        x = torch.randn(batch_size, seq_len, hidden_dim * 2, device='cuda', requires_grad=True)
        
        ref_output = geglu_reference_forward(x)
        
        x_chunks = torch.chunk(x, 2, dim=-1)
        gate, value = x_chunks[0], x_chunks[1]
        gate_flat = gate.reshape(-1)
        value_flat = value.reshape(-1)
        
        output_flat = torch.empty_like(gate_flat)
        
        n_elements = gate_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        forward_kernel[grid](gate_flat, value_flat, output_flat, n_elements, BLOCK_SIZE=1024)
        
        our_output = output_flat.reshape(gate.shape)
        
        max_diff = torch.max(torch.abs(ref_output - our_output))
        print(f"Max difference in {implementation_type} GEGLU forward pass: {max_diff.item()}")
        assert max_diff < 1e-2 if use_approx else 1e-5, f"{implementation_type} GEGLU forward pass implementation is incorrect!"
        return True
    
    def test_backward():
        """Test the backward pass of GEGLU"""
        print(f"Testing {implementation_type} GEGLU backward pass...")
        
        batch_size, seq_len, hidden_dim = 2, 10, 128
        x = torch.randn(batch_size, seq_len, hidden_dim * 2, device='cuda', requires_grad=True)
        
        x_ref = x.clone().detach().requires_grad_(True)
        ref_output = geglu_reference_forward(x_ref)
        
        grad_output = torch.randn_like(ref_output)
        
        ref_output.backward(grad_output)
        ref_grad = x_ref.grad.clone()
        
        x_chunks = torch.chunk(x, 2, dim=-1)
        gate, value = x_chunks[0], x_chunks[1]
        gate_flat = gate.reshape(-1)
        value_flat = value.reshape(-1)
        
        output_flat = torch.empty_like(gate_flat)
        n_elements = gate_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        forward_kernel[grid](gate_flat, value_flat, output_flat, n_elements, BLOCK_SIZE=1024)
        
        grad_output_flat = grad_output.reshape(-1)
        
        dW = grad_output_flat.clone()
        e = gate_flat.clone()
        g = value_flat.clone()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        backward_kernel[grid](dW, e, g, n_elements, BLOCK_SIZE=1024)
        
        our_grad = torch.cat([e.reshape(gate.shape), g.reshape(value.shape)], dim=-1)
        
        max_diff = torch.max(torch.abs(ref_grad - our_grad))
        print(f"Max difference in {implementation_type} GEGLU backward pass: {max_diff.item()}")
        assert max_diff < 1e-2 if use_approx else 1e-5, f"{implementation_type} GEGLU backward pass implementation is incorrect!"
        return True
    
    forward_passed = test_forward()
    backward_passed = test_backward()
    
    if forward_passed and backward_passed:
        print(f"All tests passed! {implementation_type.capitalize()} GEGLU implementation is correct.")
    else:
        print(f"Tests failed! {implementation_type.capitalize()} GEGLU implementation needs fixing.")

if __name__ == "__main__":
    # Test exact implementation
    test_geglu_correctness(use_approx=False)
    
    # Test approximate implementation
    test_geglu_correctness(use_approx=True)
