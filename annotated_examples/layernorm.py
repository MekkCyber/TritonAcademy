# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Modifications Copyright 2025 Mekkcyber.  
#
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
# limitations under the License

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
def layernorm_forward(
    Y, Y_row_stride,             # Output tensor and its row stride
    X, X_row_stride,             # Input tensor and its row stride
    weight,                      # Scale parameter for normalization
    bias,                        # Bias parameter for normalization
    inv_var,                     # Buffer to store inverse variance
    mean,                        # Buffer to store mean
    n_cols, eps,                 # Number of columns and epsilon for numerical stability
    BLOCK_SIZE : tl.constexpr    # Compile-time constant for block size
):
    """
    This kernel implements the forward pass of Layer Normalization using Triton.
    
    Layer Normalization normalizes each input row independently using the formula:
    y = ((x - mean) / sqrt(variance + eps)) * weight + bias
    
    Example with a 3x5 input matrix X:
    X = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # Row 0
        [6.0, 7.0, 8.0, 9.0, 10.0], # Row 1
        [11.0, 12.0, 13.0, 14.0, 15.0] # Row 2
    ]
    weight = [0.5, 0.5, 0.5, 0.5, 0.5]
    bias = [0.1, 0.1, 0.1, 0.1, 0.1]
    
    For row_idx = 1 (second CUDA thread block):
    """
    
    # Each CUDA thread block processes one row of the input
    row_idx = tl.program_id(0)
    
    # Create column indices [0, 1, 2, ..., BLOCK_SIZE-1]
    # BLOCK_SIZE is the nearest power of 2 greater than or equal to n_cols
    # These will be used to access elements within a row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle cases where n_cols < BLOCK_SIZE
    # For example, if n_cols=5 and BLOCK_SIZE=8, only the first 5 elements are valid
    mask = col_offsets < n_cols

    # In the case of Layer Normalization, the input tensor X and output tensor Y have the same shape.
    # This means we can use the same indexing pattern to access corresponding elements in both tensors.
    # We're using row_idx to determine which row we're processing, and then using the same
    # col_offsets within that row to access individual elements.
    # 
    # The row_stride parameters (X_row_stride and Y_row_stride) tell us how far to jump in memory
    # to move from one row to the next. While these are often the same for X and Y, having separate
    # stride parameters allows for flexibility in memory layout.

    # In row-major order, elements in a row are stored contiguously in memory
    # For a matrix with n_cols columns, the row_stride equals n_cols
    # Example with our 3x5 matrix X stored in row-major order in memory:
    # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    # |--------- Row0 ---------|---------- Row1 ---------|---------- Row2 -----------|
    # To move from the start of Row 0 to the start of Row 1, we add row_stride (5)
    
    # In the beginning X and Y point to the first element of their first row
    # if row_idx==1 :
    # - Y + row_idx * Y_row_stride = Y + 1 * 5 = Y + 5 points to the second row of Y
    # - X + row_idx * X_row_stride = X + 1 * 5 = X + 5 points to the second row of X

    Y  += row_idx * Y_row_stride
    X  += row_idx * X_row_stride
    # inv_var and mean are 1D tensors with n_rows elements
    # when row_idx==1, inv_var points to the second element in the inverse variance buffer
    # when row_idx==1, mean points to the second element in the mean buffer
    inv_var += row_idx
    mean += row_idx

    # Load the entire row from input tensor X
    # For row_idx=1: X_row = [6.0, 7.0, 8.0, 9.0, 10.0, 0, 0, 0]
    # The 'other=0' parameter sets values outside the mask to 0
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    
    # Load weight parameters for this row
    # weight_row = [0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0] (if BLOCK_SIZE=8)
    weight_row = tl.load(weight + col_offsets, mask = mask, other = 0).to(tl.float32)
    
    # Load bias parameters for this row
    # bias_row = [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0] (if BLOCK_SIZE=8)
    bias_row = tl.load(bias + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Calculate mean of the row
    # For row_idx=1: mean_X = (6.0 + 7.0 + 8.0 + 9.0 + 10.0) / 5 = 8.0
    mean_X = tl.sum(X_row, axis = 0) / n_cols
    
    # Subtract mean from each element in the row
    # For row_idx=1: XX = [-2.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0]
    XX = tl.where(mask, X_row - mean_X, 0)
    
    # Calculate variance of the row
    # For row_idx=1: row_var = ((-2.0)² + (-1.0)² + 0.0² + 1.0² + 2.0²) / 5 = (4.0 + 1.0 + 0.0 + 1.0 + 4.0) / 5 = 2.0
    row_var = tl.sum(XX * XX, axis = 0) / n_cols
    
    # Calculate inverse square root of variance (for stability, add epsilon)
    # For row_idx=1: inv_var_val = 1/sqrt(2.0 + eps) ≈ 0.707
    inv_var_val = tl.math.rsqrt(row_var + eps)
    
    # Store the inverse variance and mean for later use in backward pass
    tl.store(inv_var, inv_var_val)
    tl.store(mean, mean_X)
    
    # Calculate normalized output with scaling and bias
    # For row_idx=1: 
    # output = ([-2.0, -1.0, 0.0, 1.0, 2.0] * 0.707) * [0.5, 0.5, 0.5, 0.5, 0.5] + [0.1, 0.1, 0.1, 0.1, 0.1]
    #        = [-0.607, -0.2535, 0.1, 0.4535, 0.807]
    output = (XX * inv_var_val) * weight_row + bias_row
    
    # Store the output row
    tl.store(Y + col_offsets, output, mask = mask)


@triton.jit
def layernorm_backward(
    dY, dY_row_stride,             # Gradient from upstream and its row stride
    X, X_row_stride,               # Input tensor and its row stride
    weight,                        # Scale parameter for normalization
    bias,                          # Bias parameter for normalization
    inv_var,                       # Stored inverse variance from forward pass
    mean,                          # Stored mean from forward pass
    n_cols, eps,                   # Number of columns and epsilon for numerical stability
    BLOCK_SIZE : tl.constexpr      # Compile-time constant for block size
):
    """
    This kernel implements the backward pass of Layer Normalization using Triton.
    
    The backward pass computes the gradient with respect to the input (dX) given the
    gradient with respect to the output (dY).
    
    Example with a 3x5 input matrix X and corresponding gradient dY:
    X = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # Row 0
        [6.0, 7.0, 8.0, 9.0, 10.0], # Row 1
        [11.0, 12.0, 13.0, 14.0, 15.0] # Row 2
    ]
    dY = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Row 0
        [0.6, 0.7, 0.8, 0.9, 1.0],  # Row 1
        [1.1, 1.2, 1.3, 1.4, 1.5]   # Row 2
    ]
    weight = [0.5, 0.5, 0.5, 0.5, 0.5]
    """
    
    # Each CUDA thread block processes one row of the input
    row_idx = tl.program_id(0)
    
    # Create column indices [0, 1, 2, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle cases where n_cols < BLOCK_SIZE
    mask = col_offsets < n_cols

    # Calculate pointers to the current row in each tensor
    # For row_idx=1, we're processing the second row of each tensor
    dY += row_idx * dY_row_stride
    X  += row_idx * X_row_stride
    inv_var += row_idx
    mean += row_idx

    # Load the gradient from upstream (dY)
    # For row_idx=1: dY_row = [0.6, 0.7, 0.8, 0.9, 1.0, 0, 0, 0]
    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    
    # Load the input values
    # For row_idx=1: X_row = [6.0, 7.0, 8.0, 9.0, 10.0, 0, 0, 0]
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    
    # Load weight parameters
    # weight_row = [0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0]
    weight_row = tl.load(weight + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Load the stored inverse variance and mean from the forward pass
    # For row_idx=1: inv_var_val ≈ 0.707, mean_val = 8.0
    inv_var_val = tl.load(inv_var).to(tl.float32)
    mean_val = tl.load(mean).to(tl.float32)
    
    # Calculate the normalized input values (same as in forward pass)
    # For row_idx=1: normed = [(6.0-8.0)*0.707, (7.0-8.0)*0.707, (8.0-8.0)*0.707, (9.0-8.0)*0.707, (10.0-8.0)*0.707]
    #                       = [-1.414, -0.707, 0.0, 0.707, 1.414]
    normed = (X_row - mean_val) * inv_var_val
    
    # Scale the upstream gradient by the weight
    # For row_idx=1: dY_W = [0.6*0.5, 0.7*0.5, 0.8*0.5, 0.9*0.5, 1.0*0.5]
    #                     = [0.3, 0.35, 0.4, 0.45, 0.5]
    dY_W = dY_row * weight_row
    
    # Calculate the gradient with respect to the input
    # This follows the chain rule for backpropagation through layer normalization
    # The formula has three terms:
    # 1. dY_W: direct contribution from upstream gradient
    # 2. -tl.sum(dY_W, axis=0)/n_cols: contribution from the mean term
    # 3. -normed * tl.sum(dY_W * normed, axis=0)/n_cols: contribution from the variance term
    
    # In general, the result would be non-zero and would then be scaled by inv_var_val
    dX_row = dY_W - tl.sum(dY_W, axis = 0) / n_cols - normed * tl.sum(dY_W * normed, axis = 0) / n_cols
    dX_row = dX_row * inv_var_val
    
    # Store the gradient with respect to the input
    # Note: We're reusing the dY tensor to store the result (in-place operation)
    tl.store(dY + col_offsets, dX_row, mask = mask)


class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device
        Y  = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        inv_var = torch.empty(n_rows, dtype = torch.float32, device = device)
        mean = torch.empty(n_rows, dtype = torch.float32, device = device)

        layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            weight,
            bias,
            inv_var,
            mean,
            n_cols, eps,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(X, weight, bias, inv_var, mean)
        return Y.view(*shape)
    pass

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, weight, bias, inv_var, mean = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X.stride(0),
            weight,
            bias,
            inv_var,
            mean,
            n_cols, ctx.eps,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None, None

def fast_layernorm(layernorm, X):
    assert(layernorm.elementwise_affine is True)
    W    = layernorm.weight
    bias = layernorm.bias
    eps = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_Layernorm.apply(X, W, bias, eps)
    return out