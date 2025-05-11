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
# limitations under the License.

import triton
import triton.language as tl
import torch
from transformers.models.llama.modeling_llama import logger

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
def _cross_entropy_forward(
    logits_ptr        ,  # Pointer to logits tensor [batch*seq_len, vocab_size]
    logits_row_stride ,  # Stride for accessing rows in logits
    loss_ptr          ,  # Pointer to output loss values
    logsumexp_ptr     ,  # Pointer to store logsumexp values (needed for backward)
    labels_ptr        ,  # Pointer to label indices
    VOCAB_SIZE        ,  # Size of vocabulary
    BLOCK_SIZE        : tl.constexpr,  # Block size for parallel processing
    DO_SOFTCAPPING    ,  # Flag for logit softcapping (e.g., for Gemma 2)
    SOFTCAP           ,  # Softcapping parameter value
    DO_LOGIT_SCALING  ,  # Flag for logit scaling (e.g., for Cohere models)
    LOGIT_SCALE       ,  # Scaling factor for logits
):
    """
    Computes cross-entropy loss in a numerically stable way.
    
    Cross Entropy Loss Formula:
        CE = -∑(y_i * log(p_i))  where p_i = softmax(x_i) = exp(x_i) / ∑exp(x_j)
    
    For one-hot labels (our case), this simplifies to:
        CE = -log(p_correct) = -(logit_correct - logsumexp)
            = logsumexp - logit_correct
    
    Numerical Stability:
    We use the LogSumExp trick for numerical stability:
        logsumexp(x) = max(x) + log(∑exp(x - max(x)))
    
    This prevents overflow by ensuring the largest exponentiated term is exp(0.0) = 1.0.
    
    Special handling:
    - If label == -100: loss = 0 (ignore token, e.g., padding)
    - Otherwise: loss = logsumexp - logit_correct
    """
    # Get current row index from the program ID, every block thread will handle a different row
    row_idx = tl.program_id(0)
    
    # Offset pointers to the current row
    # we cast to tl.int64 to avoid overflow because vocab sizes are large
    logits_ptr    += row_idx * triton_cast(logits_row_stride, tl.int64)
    # each row corresponds to a different token in the sequence, hence a loss, logsumexp and label
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx
    labels_ptr    += row_idx

    # Create offsets for accessing columns in parallel
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Create mask for valid vocabulary indices
    mask = col_offsets < VOCAB_SIZE

    # Load the label index for this row
    label_idx = tl.load(labels_ptr).to(tl.int32)
    # Load logits for this row, masking invalid indices
    # we mask invalid indices to -infinity to ensure they don't contribute to the sum (exp(-infinity) = 0)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

    # Apply logit scaling if enabled: x → t*x (t = LOGIT_SCALE)
    # This scales the logits before softmax, affecting the "temperature" of the distribution
    # Higher values (t > 1) make the distribution more uniform/smoother
    # Lower values (0 < t < 1) make the distribution more peaked/confident
    # Logit scaling was introduced in models like Cohere Command and Claude to control
    # the model's confidence in its predictions. It helps prevent overconfidence and
    # can improve model calibration, especially in out-of-distribution scenarios.
    # Unlike temperature sampling at inference time, this scaling is applied during training.
    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    
    # Apply logit softcapping if enabled: x → t*tanh(x/t) (t = SOFTCAP)
    # This bounds logits to [-t, t] range, preventing extreme values
    # Softcapping was introduced in models like Gemma 2 to improve training stability
    # by preventing logits from growing too large, which can cause:
    #   1. Numerical instability in softmax computation
    #   2. Overconfident predictions leading to poor generalization
    #   3. Gradient explosion during backpropagation
    # Unlike simple clipping, tanh-based softcapping maintains differentiability
    # and allows gradients to flow even for extreme values, just at a reduced magnitude.
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    
    # Compute logsumexp in a numerically stable way
    # First find the maximum logit value
    c = tl.max(logits, 0)
    # Then compute logsumexp = max + log(sum(exp(logits - max)))
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    # Compute loss only if label is valid (not -100)
    if label_idx != -100:
        # Load the logit for the correct class
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        
        # Apply the same transformations to the target logit
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:   x = SOFTCAP * triton_tanh(x / SOFTCAP)
        
        # Compute cross entropy: logsumexp - correct_logit
        # This is equivalent to -log(softmax(correct_logit))
        loss = logsumexp - x
    else:
        # For padding tokens (label_idx == -100), set loss to 0
        loss = 0.0
        
    # Store results for this row
    tl.store(logsumexp_ptr, logsumexp)  # Save logsumexp for backward pass
    tl.store(loss_ptr, loss)            # Save the computed loss

@triton.jit
def _cross_entropy_backward(
    logits_ptr        ,  # Pointer to input logits
    logits_row_stride ,  # Stride between rows in logits tensor
    dloss_ptr         ,  # Pointer to gradient of loss w.r.t output
    dloss_row_stride  ,  # Stride between rows in dloss tensor
    logsumexp_ptr     ,  # Pointer to precomputed logsumexp values
    labels_ptr        ,  # Pointer to target labels
    VOCAB_SIZE        ,  # Size of vocabulary (number of classes)
    BLOCK_SIZE        : tl.constexpr,  # Size of processing block
    DO_SOFTCAPPING    ,  # Whether to apply softcapping
    SOFTCAP           ,  # Softcapping parameter value
    DO_LOGIT_SCALING  ,  # Whether to apply logit scaling
    LOGIT_SCALE       ,  # Logit scaling parameter value
):
    """
    Backward pass for cross entropy loss.
    
    Cross Entropy Loss: CE(x, class) = -log(softmax(x)[class])
                                     = -log(exp(x_class) / sum(exp(x_i)))
                                     = -x_class + log(sum(exp(x_i)))
    
    For the backward pass, we need to compute gradients w.r.t. each logit.
    
    Let L = CE(x, class) and z = log(sum(exp(x_i))) (logsumexp)
    
    For the correct class (i = class):
        dL/dx_class = d/dx_class(-x_class + z) = -1 + exp(x_class - z) = -1 + softmax(x_class) (check backprop_math/cross_entropy.md)
    
    For other classes (i ≠ class):
        dL/dx_i = d/dx_i(-x_class + z) = d/dx_i(z) = exp(x_i - z) = softmax(x_i) (check backprop_math/cross_entropy.md)
    
    When logit transformations are applied, we use the chain rule to compute gradients.
    """
    # Get current row and block indices
    row_idx   = tl.program_id(0)

    # Calculate pointers for current row
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    dloss_ptr  += row_idx * dloss_row_stride
    
    # Calculate column offsets for current block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Create mask for valid vocabulary indices
    mask = col_offsets < VOCAB_SIZE
    
    # Load the target label for current row
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    # Load gradient of loss w.r.t output
    # For padding tokens (label_idx == -100), set gradient to 0
    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    # Load logits for current row
    x = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

    # Apply logit scaling if enabled
    # If x is scaled as x' = s*x in forward, then dx'/dx = s
    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    # Store original values before softcapping for gradient calculation
    # For softcapping: x' = t*tanh(x/t), we need to track intermediate values they will be used in the backward pass chain rule
    tanh_term = x
    if DO_SOFTCAPPING:
        # Apply softcapping: x' = t*tanh(x/t)
        tanh_term = triton_tanh(x / SOFTCAP)  # Store tanh(x/t) for gradient calculation
        x = SOFTCAP * tanh_term  # This is the softcapped value

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    
    # Compute softmax: exp(x - logsumexp) = softmax(x) for the whole row
    # This gives us part of the gradient formula
    y = tl.exp(x - logsumexp)
    
    # Adjust gradient for the target class
    # For i = target: gradient = softmax(x_i) - 1
    # For i ≠ target: gradient = softmax(x_i)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,  # For target class: exp(x - logsumexp) - 1
        y,        # For other classes: exp(x - logsumexp)
    )

    # Apply chain rule for logit scaling
    # If x' = s*x, then dL/dx = dL/dx' * dx'/dx = dL/dx' * s
    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    # Apply chain rule for softcapping
    # For x' = t*tanh(x/t), dx'/dx = 1 - tanh²(x/t)
    # This is the derivative of tanh: d/dx[tanh(x)] = 1 - tanh²(x)
    if DO_SOFTCAPPING:
        y = y * (1.0 - tanh_term*tanh_term)  # tanh_term = tanh(x/t)

    # Store final gradients
    # For padding tokens (label_idx == -100), gradient is 0
    tl.store(logits_ptr + col_offsets, dloss * y, mask = mask)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping : float = 0, logit_scaling : float = 0):
        n_rows : int
        vocab_size : int
        n_rows, vocab_size = logits.shape

        losses = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

        DO_SOFTCAPPING   : bool = bool(logit_softcapping != 0)
        DO_LOGIT_SCALING : bool = bool(logit_scaling != 0)

        BLOCK_SIZE : int
        num_warps  : int
        # For small vocabs <= 65336 like Llama, Mistral
        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
        logsumexp = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

        _cross_entropy_forward[(n_rows,)](
            logits, logits.stride(0),
            losses,
            logsumexp,
            labels,
            VOCAB_SIZE       = vocab_size,
            BLOCK_SIZE       = BLOCK_SIZE,
            DO_SOFTCAPPING   = DO_SOFTCAPPING,
            SOFTCAP          = logit_softcapping,
            DO_LOGIT_SCALING = DO_LOGIT_SCALING,
            LOGIT_SCALE      = logit_scaling,
            num_warps        = num_warps,
        )

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING    = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING  = DO_LOGIT_SCALING
        ctx.logit_scaling     = logit_scaling
        return losses
    pass


    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows : int
        vocab_size : int
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)

        _cross_entropy_backward[(n_rows,)](
            logits,   logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE       = vocab_size,
            BLOCK_SIZE       = BLOCK_SIZE,
            DO_SOFTCAPPING   = ctx.DO_SOFTCAPPING,
            SOFTCAP          = ctx.logit_softcapping,
            DO_LOGIT_SCALING = ctx.DO_LOGIT_SCALING,
            LOGIT_SCALE      = ctx.logit_scaling,
            num_warps        = num_warps,
        )
        return logits, None, None, None


def fast_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items


def reference_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0):
    """Reference implementation using PyTorch's native functions"""
    if logit_scaling != 0:
        logits = logits * logit_scaling
    
    if logit_softcapping != 0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Get the log probability for the correct labels
    label_mask = labels != -100
    labels_masked = labels.clone()
    labels_masked[~label_mask] = 0 
    
    # Gather the log probabilities for the correct labels
    label_log_probs = log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)
    
    # Apply the mask to ignore padding tokens
    label_log_probs = label_log_probs * label_mask
    
    loss = -label_log_probs.sum() / label_mask.sum()
    return loss


def test_cross_entropy():
    """Test the forward and backward pass of the custom cross entropy implementation"""
    print("Testing Fast Cross Entropy implementation...")
    
    # Test configurations
    test_configs = [
        {"name": "Standard", "softcap": 0, "scaling": 0},
        {"name": "With Softcapping", "softcap": 10.0, "scaling": 0},
        {"name": "With Scaling", "softcap": 0, "scaling": 2.0},
        {"name": "With Both", "softcap": 10.0, "scaling": 2.0}
    ]
    
    for config in test_configs:
        print(f"\nTesting {config['name']} configuration...")
        
        # Create test inputs
        batch_size, seq_len, vocab_size = 2, 10, 32000
        logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', requires_grad=True)
        # Create labels with some -100 values to test padding
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        labels[0, 0] = -100  # Add some padding tokens
        
        # Clone inputs for reference implementation
        logits_ref = logits.clone().detach().requires_grad_(True)
        
        # Forward pass
        our_loss = fast_cross_entropy_loss(
            logits, labels, 
            logit_softcapping=config['softcap'], 
            logit_scaling=config['scaling']
        )
        
        # Reference implementation
        ref_loss = reference_cross_entropy_loss(
            logits_ref, labels,
            logit_softcapping=config['softcap'],
            logit_scaling=config['scaling']
        )
        
        # Compare forward results
        forward_diff = torch.abs(our_loss - ref_loss).item()
        print(f"Forward pass difference: {forward_diff:.6f}")
        assert forward_diff < 1e-4, f"Forward pass failed for {config['name']} configuration!"
        
        # Backward pass
        our_loss.backward()
        ref_loss.backward()
        # Compare gradients

        grad_diff = torch.max(torch.abs(logits.grad - logits_ref.grad)).item()
        print(f"Max gradient difference: {grad_diff:.6f}")
        assert grad_diff < 1e-4, f"Backward pass failed for {config['name']} configuration!"
        
        # Reset gradients for next test
        logits.grad.zero_()
        logits_ref.grad.zero_()
    
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    test_cross_entropy()
