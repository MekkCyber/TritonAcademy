# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# Modified by MekkCyber - 2025
#********************************************************
import triton
import triton.language as tl

@triton.jit
def dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample: tl.constexpr, W_group_mode: tl.constexpr, zero_is_scalar: tl.constexpr):
    """
    Dequantizes packed integer values into floating point values using various quantization schemes.
    
    Args:
        b: Packed quantized values (typically int32)
        scales: Scaling factors for dequantization (per group or channel)
        zeros: Zero points for asymmetric quantization (per group or channel)
        q_shift: Bit shift amount for unpacking elements from packed format
        meta_dtype: Target data type for metadata operations
        unpack_mask: Bit mask for extracting individual elements (e.g., 0xF for 4-bit)
        elements_per_sample: Number of quantized elements packed into each storage unit
        W_group_mode: Quantization scheme to use (1-4)
        zero_is_scalar: Whether zero point is shared across all elements
    
    Returns:
        Dequantized tensor in floating point format
    """
    # Step 1: Unpack the elements if they are packed (e.g., 8 4-bit values in one int32)
    if(elements_per_sample > 1):
        # Extract individual quantized values using bit shifting and masking
        # q_shift determines which element to extract based on position
        b = (b >> q_shift) & unpack_mask # int32 -> int32

    # Step 2: Apply the appropriate dequantization formula based on W_group_mode
    
    if(W_group_mode == 1): # Shift-only mode (zero-point subtraction)
        # Formula: dequantized = quantized - zero_point
        b = b.to(meta_dtype) - zeros 

    if(W_group_mode == 2): # Scale-only mode (symmetric quantization)
        # Formula: dequantized = quantized * scale
        # Used when quantized values are centered around zero
        b = b.to(meta_dtype) * scales

    if(W_group_mode == 3): # Scale and shift mode (asymmetric quantization)
        # Formula: dequantized = (quantized - zero_point) * scale
        if(zero_is_scalar):
            # When zero_point is shared across all elements (memory optimization)
            b = (b - zeros).to(meta_dtype) * scales
        else:
            # When each group has its own zero_point
            b = (b.to(meta_dtype) - zeros) * scales

    if(W_group_mode == 4): # Fused multiply-add mode
        # Formula: dequantized = quantized * scale + zero
        # Uses fused multiply-add for better performance
        # Note: in this mode, 'zeros' is actually an additive term, not a zero point
        b = tl.fma(b.to(meta_dtype), scales, zeros)

    return b

@triton.jit
def swizzle_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_n
    group_id   = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    return pid_m, pid_n


@triton.jit
def gemm_splitK_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K,
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_g, stride_meta_n,
    ######### Dtypes #########
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr, atomic_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
):
    """
    Quantized GEMM with split-K parallelism for C = matmul(A, dequantize(B, scales, zeros))
    
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 (same dtype as input A)
    scales is of shape (K//group_size, N) or (1, N): meta_dtype
    zeros is of shape (K//group_size, N) or (1, 1): meta_dtype
    
    Requirements:
    - BLOCK_SIZE_M must be >= 16
    - BLOCK_SIZE_K * SPLIT_K must be <= group_size
    
    Based on the split-K dequantization GEMM implementation from:
    https://github.com/foundation-model-stack/foundation-model-stack/blob/main/fms/models/llm/kernels/triton/splitk_dequant_gemm.py
    """

    # It's recommended to understand the standard GEMM implementation first in gemlite/gemm.py, as this split-K version
    # builds upon it, so we will not delve into the details of the standard GEMM implementation here.
    #
    # Unlike standard GEMM where each thread block computes its entire output tile,
    # in split-K GEMM we have a 2D grid of thread blocks where:
    # - pid (x-dim): Each thread block along x-axis is responsible for a unique output tile
    # - pid_k (y-dim): Multiple thread blocks along y-axis collaborate on the same output tile,
    #   each computing a partial result and using atomic adds to safely accumulate into the final output
    pid   = tl.program_id(axis=0)  # Determines which output tile to compute
    pid_k = tl.program_id(axis=1)  # Determines which K-slice to process for this output tile

    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) 

    offs_am = offs_m
    offs_ak = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)

    if(data_contiguous):
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
        offs_bk = offs_k
    else:
        offs_bn = offs_n
        offs_bk = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)

    b_ptrs  = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 
    q_shift = ((offs_bk % elements_per_sample) * W_nbits).to(tl.int32)[:, None] 

    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = offs_am[:, None] < M
    
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n

    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size

    # BLOCK_SIZE_K_U: How much to advance pointers in matrix A (unpacked matrix)
    # We multiply by SPLIT_K since each thread block processes only K / SPLIT_K * BLOCK_SIZE_K elements 
    # This represents the stride in the K dimension for matrix A
    BLOCK_SIZE_K_U: tl.constexpr = BLOCK_SIZE_K   * SPLIT_K

    # BLOCK_SIZE_K_P: How much to advance pointers in matrix B (packed matrix) 
    # Since B is packed with elements_per_sample values per int32
    # We divide BLOCK_SIZE_K by elements_per_sample to get number of int32s
    # Then multiply by SPLIT_K for the same reason as above
    # This represents the stride in the K dimension for packed matrix B
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample) * SPLIT_K

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(num_pid_k):

        if(A_load_order == 0):
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 

        b = tl.load(b_ptrs, eviction_policy='evict_first')

        if(A_load_order == 1): 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 
        
        # This code calculates which group we're currently processing for weight quantization
        if(W_group_mode > 0):
            # Important: We need BLOCK_SIZE_K to be smaller than group_size
            # This is because we only load one line from the scales here
            # 
            # Example with proper sizing:
            # - BLOCK_SIZE_K = 8 (how many K elements we process per block)
            # - group_size = 32 (we quantize weights in groups of 32)
            # - SPLIT_K = 4 (we split K dimension across 4 blocks)
            # - stride_mul = BLOCK_SIZE_K/group_size = 8/32 = 0.25 (fraction of a group in one block)
            #
            # k: outer loop counter (0, 1, 2..., tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K))
            # pid_k: which split the thread block is responsible for (0 to SPLIT_K-1)
            #
            # For example, if k=2, SPLIT_K=4, pid_k=1:
            # k * SPLIT_K + pid_k = 2 * 4 + 1 = 9
            # 
            # Multiply by stride_mul to convert from blocks to groups:
            # 9 * 0.25 = 2.25, which means we're processing part of the 2nd group
            k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32)

        if(W_group_mode >= 2): #[2, 3, 4]
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            scales = None

        if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
            if(zero_is_scalar):
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs  + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            zeros = None

        if(A_load_order == 2): #Mid load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if(A_load_order == 3):
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32") 
        
        # Advance pointers for the next iteration of the k-loop
        a_ptrs += BLOCK_SIZE_K_U * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk

    if(channel_scale_mode == 1):
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2):
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3):
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr   + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    acc = acc.to(output_dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    # We know that each thread block computes a partial result for the same output location (M,N coordinates)
    # When SPLIT_K > 1, multiple blocks will write to the same memory location
    # We use atomic_add to safely accumulate these partial results from different blocks
    # without race conditions, ensuring all contributions are correctly summed
    # The atomic operation guarantees that concurrent updates to the same memory
    # location happen in a coordinated way, preventing data corruption
    if(SPLIT_K > 1):
        tl.atomic_add(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N), sem=atomic_mode) #release / relaxed
    else:
        # When SPLIT_K = 1, each output location is computed by exactly one block
        # so we can use a simple store operation instead of an atomic add (this is the same as the standard GEMM)
        tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)) 