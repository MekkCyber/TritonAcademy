# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# Modified by Mekkcyber - 2025
#********************************************************
import torch, time
import triton
import triton.language as tl

# Prerequisities for the gemm, for better understanding start from the gemm_kernel function, everything is explained there
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

# Main kernel, that should be studied for better understanding
@triton.jit
def gemm_kernel(
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
    GROUP_SIZE_M: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr,
    data_contiguous: tl.constexpr,
):
    """
    Based on https://github.com/fpgaminer/GPTQ-triton
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K <= group_size
    """
    
    # This kernel implements a quantized matrix multiplication operation where:
    # - Matrix A is a full-precision matrix (float16/bfloat16)
    # - Matrix B is a quantized matrix (packed into int32)
    # - The result C = A @ dequantize(B)
    
    # Example:
    # If we have:
    # - A: 128x512 matrix (M=128, K=512) in float16
    # - B: quantized 128x256 matrix packed into int32
    # - W_nbits=4 (4-bit quantization)
    # - elements_per_sample=8 (8 elements packed into each int32)
    # B should be dequantized to a 512x256 matrix in float16
    # Then C will be a 128x256 matrix in float16
    
    # Get the program ID which identifies which tile this thread block processes
    pid = tl.program_id(axis=0)
    
    # for each pid, we need to find the corresponding block in the output matrix, we can do this in two ways:
    # 1. swizzle_tile: Creating groups horizontally and inside each group, pids are mapped vertically
    #    Example of swizzle pattern with GROUP_SIZE_M=2:
    #    pid layout in the output matrix (each number represents a block's pid):
    #    0  2  4  6  
    #    1  3  5  7 
    #    8  10 12 14
    #    9  11 13 15
    #    This improves cache locality by keeping adjacent thread blocks working on adjacent rows, see classic/matmul.py for more details
    # 2. linear_tile: we simply set pid_m = pid // (tl.cdiv(N, BLOCK_SIZE_N)) and pid_n = pid % (tl.cdiv(N, BLOCK_SIZE_N))
    #    Example of linear pattern with GROUP_SIZE_M=2:
    #    pid layout in the output matrix (each number represents a block's pid):
    #    0  1  2  3
    #    4  5  6  7
    #    8  9  10 11
    #    12 13 14 15
    #    This is simpler to compute but may result in poorer cache locality
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Calculate how many blocks we need in the K dimension
    # For example, if K=512 and BLOCK_SIZE_K=64, we need 8 iterations
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Calculate offsets for each thread within the block
    # If BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, each thread block processes a 16x16 tile
    # offs_m will be [0,1,2,...,15] + (pid_m * 16)
    # offs_n will be [0,1,2,...,15] + (pid_n * 16)
    # offs_k will be [0,12,...,BLOCK_SIZE_K-1]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Optimize memory access patterns based on data layout
    if(data_contiguous):
        # multiple_of(tensor, values): Informs the compiler that the shape dimensions of the tensor are multiples of 'values'
        # - This allows the compiler to optimize memory access patterns and vectorize loads
        # - For example, if the shape of offs_am is a multiple of BLOCK_SIZE_M, the compiler can generate more efficient code
        #
        # max_contiguous(tensor, values): Informs the compiler that the first 'values' elements in the tensor are contiguous
        # - This helps the compiler generate more efficient memory access patterns
        # - For example, if the first BLOCK_SIZE_M elements in offs_am are contiguous (0,1,2,...), 
        #   the compiler can use coalesced memory accesses
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    
    # Calculate pointers to input matrices
    # For matrix A: If stride_am=K and stride_ak=1, this accesses A in row-major order
    # Example: For A[2,3] with K=512, the offset would be 2*512 + 3 = 1027
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    # Create a mask to handle boundary conditions (when M is not divisible by BLOCK_SIZE_M)
    a_mask = (offs_am[:, None] < M)
    
    # For matrix B: Calculate pointers based on the packed format
    # If B is packed with 8 elements per int32, we divide offs_k by 8 to get the actual memory location
    # 
    # Example: With 8-bit elements packed into 32-bit integers (elements_per_sample = 4):
    # - For offs_k values [0,1,2,3], the division gives [0,0,0,0]
    # - For offs_k values [4,5,6,7], the division gives [1,1,1,1]
    #  
    # This means that elements 0-3 are stored in the first 32-bit word, elements 4-7 in the second word, etc.
    # Later, we'll use q_shift to extract the correct bits from each packed word.
    b_ptrs = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 

    # Calculate bit shift for unpacking quantized values from packed integers
    # 
    # Example 1: With 4-bit quantization (W_nbits=4) and 8 elements per int32 (elements_per_sample=8):
    # - For offs_k = [0,1,2,3,4,5,6,7]:
    #   offs_k % elements_per_sample = [0,1,2,3,4,5,6,7]
    #   q_shift = [0,4,8,12,16,20,24,28] bits
    # 
    # Example 2: With 8-bit quantization (W_nbits=8) and 4 elements per int32 (elements_per_sample=4):
    # - For offs_k = [0,1,2,3,4,5,6,7]:
    #   offs_k % elements_per_sample = [0,1,2,3,0,1,2,3]
    #   q_shift = [0,8,16,24,0,8,16,24] bits
    #
    # The modulo operation (%) wraps around when we exceed elements_per_sample,
    # ensuring we extract the correct element position within each packed integer.
    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    
    # Calculate pointers to quantization metadata (scales and zeros)
    # These pointers point to the start of each column in the metadata matrices
    # For example, if we have a matrix with N columns, each column has its own
    # scale and zero point values for dequantization
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_meta_n
    
    # Calculate stride multiplier for group quantization
    # If group_size=64 and BLOCK_SIZE_K=32, stride_mul=0.5
    # This means we need a new scale/zero for every 2 K-dimension blocks
    stride_mul = BLOCK_SIZE_K / group_size 

    # If zero point is a scalar (same for all elements), load it once
    # eviction_policy='evict_last' tells the compiler how to manage the cache:
    # - 'evict_last': Keep the data in cache as long as possible (good for reused data)
    # - 'evict_first': Evict from cache quickly (good for data used only once)
    # This helps optimize memory access patterns and cache utilization
    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

    # Initialize accumulator for matrix multiplication
    # This will store the partial sums during the computation
    # For a 16x16 tile, this is a 16x16 matrix initialized to zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype) 

    # Main computation loop - iterate over blocks in the K dimension
    # For example, if K=512 and BLOCK_SIZE_K=64, we do 8 iterations
    for k in range(num_pid_k):
        # Load matrix A based on the specified loading order
        # Different loading orders can help with instruction scheduling and can lead to better performance
        if(A_load_order == 0): # Early load - load A before B
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        # Load packed quantized values from matrix B
        # Load packed quantized weights with 'evict_first' policy since we'll immediately
        # dequantize these values and won't need the packed representation in cache
        # Each row of B is repeated elements_per_sample times, this way we can unpack it using the q_shift
        # If you don't get why you can look at how b_ptrs is computed
        b = tl.load(b_ptrs, eviction_policy='evict_first')

        if(A_load_order == 1): # Load A after loading B
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        # Load quantization metadata (scales and zero points) based on the group mode
        # Different modes use different patterns of scales and zeros
        # W_group_mode controls how quantization parameters (scales and zeros) are applied:
        
        # Mode 0: No quantization - neither scales nor zeros are used
        # Mode 1: Zero-point only quantization - only zero points are used, no scales
        #         Used for integer quantization where only zero-point shifting is needed
        # Mode 2: Scale-only quantization - only scales are used, no zero points
        #         Used for symmetric quantization where values are centered around zero
        # Mode 3: Full quantization - both scales and zero points are used
        #         Used for asymmetric quantization with arbitrary ranges
        # Mode 4: Asymmetric (Grouped - b*scales + zeros)
        
        if(W_group_mode > 0):          
            # Calculate offset for grouped quantization
            # For every group_size weights, we have a single scale/zero point
            # stride_mul = BLOCK_SIZE_K / group_size controls how often we need new metadata
            # 
            # Examples:
            # 1. If group_size=64 and BLOCK_SIZE_K=64, stride_mul=1
            #    We need a new scale for each K block (k_m increases by 1 each iteration)
            # 2. If group_size=128 and BLOCK_SIZE_K=64, stride_mul=0.5
            #    We need a new scale every 2 K blocks (k_m increases by 1 every 2 iterations)
            #
            # This mapping ensures we use the correct scale/zero for each weight group
            k_m = (k * stride_mul).to(tl.int32)

        # Load scales if needed (modes 2, 3, 4)
        # Example: For per-channel quantization, each output channel has its own scale
        if(W_group_mode >= 2): # [2, 3, 4]
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            scales = None

        # Load zero points if needed (modes 1, 3, 4)
        # Example: For per-channel quantization, each output channel has its own zero point
        if(W_group_mode == 1 or W_group_mode >= 3): # [1, 3, 4]
            if(zero_is_scalar):
                # If zero_is_scalar=1, use the same zero point for all elements
                # This saves memory and bandwidth when all channels share the same zero point
                zeros = zero_scalar
            else:
                # Otherwise load per-group zero points from memory
                # stride_meta_g controls the spacing between groups in memory
                zeros = tl.load(zeros_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            zeros = None

        if(A_load_order == 2): # Mid load - load A after loading metadata
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        # Unpack and dequantize the values from matrix B
        # Dequantization formula depends on the W_group_mode:
        # - Mode 0: No dequantization just unpacking
        # - Mode 1: dequantized_value = quantized_value - zero_point
        # - Mode 2: dequantized_value = quantized_value * scale
        # - Mode 3: dequantized_value = (quantized_value - zero_point) * scale
        # - Mode 4: dequantized_value = b*scales + zeros
        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if(A_load_order == 3): # Late load - load A after dequantization
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        # Perform matrix multiplication for this block
        # For 16x16 tiles, this computes a 16x16 result and adds to accumulator
        # Example: If a is 16x64 and b is 64x16, this computes a 16x16 partial result
        acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32")

        # Advance pointers for the next iteration
        # Move to the next block in the K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

    # Apply channel-wise scaling to the result if needed
    # This is used for various quantization schemes
    if(channel_scale_mode == 1): # Weight-only scaling
        # Load scales for each output channel
        # Example: If each output has a different scale factor
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        # Apply scales to each column of the result
        acc = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2): # Activation-only scaling
        # Load scaling factors for each input channel (row of the activation matrix)
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        
        # Create a vector of ones for the output dimension
        # Since we're only scaling by activation (input channels), we use 1.0 for all output channels
        # This creates a vector of size BLOCK_SIZE_N filled with 1's of the metadata type
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        
        # Apply the scaling factors to the accumulated result:
        # 1. scales_a[:, None]: Reshape scales_a from [BLOCK_SIZE_M] to [BLOCK_SIZE_M, 1] for broadcasting
        # 2. scales_b[None, :]: Reshape scales_b from [BLOCK_SIZE_N] to [1, BLOCK_SIZE_N] for broadcasting
        # 3. This creates a scaling matrix of shape [BLOCK_SIZE_M, BLOCK_SIZE_N] where each row is scaled by its corresponding scales_a value
        # 4. Multiply the accumulator by this scaling matrix element-wise
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3): # Both weight and activation scaling
        # Load scales for both input and output channels
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        # Apply both scales to the result
        # Example: If row 2 has scale 0.5 and column 3 has scale 0.25, 
        # element [2,3] is multiplied by 0.5*0.25=0.125
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    # Convert the result to the output data type
    acc = acc.to(output_dtype)

    # Calculate pointers to the output matrix
    # Similar to input pointers, but for matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Store the result to the output matrix
    # Use masks to handle boundary conditions
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)) 


def test_kernel():
    
    # Set up test parameters
    M, N, K = 128, 256, 512
    W_nbits = 4
    group_size = 64
    elements_per_sample = 8  # For 4-bit quantization, 8 elements fit in one int32
    
    # Create input matrices
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # Create quantized weights (normally this would come from a quantization process)
    # For testing, we'll create random data
    b_unpacked = torch.randint(-8, 7, (K, N), dtype=torch.int8, device='cuda')
    
    # Pack the weights into int32
    b_packed = torch.zeros((K // elements_per_sample, N), dtype=torch.int32, device='cuda')
    for i in range(elements_per_sample):
        b_packed |= (b_unpacked[i::elements_per_sample, :].to(torch.int32) & ((1 << W_nbits) - 1)) << (i * W_nbits)
    
    # Create scales and zeros for dequantization
    scales = torch.ones((K // group_size, N), dtype=torch.float16, device='cuda')
    zeros = torch.zeros((K // group_size, N), dtype=torch.float16, device='cuda')
    scales_a = torch.ones(M, dtype=torch.float16, device='cuda')
    
    # Output matrix
    c_triton = torch.zeros((M, N), dtype=torch.float16, device='cuda')
    
    # Calculate strides
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b_packed.stride(0), b_packed.stride(1)
    stride_cm, stride_cn = c_triton.stride(0), c_triton.stride(1)
    stride_meta_g, stride_meta_n = scales.stride(0), scales.stride(1)
    
    # Define grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Run the kernel
    gemm_kernel[grid](
        a_ptr=a, b_ptr=b_packed, c_ptr=c_triton,
        scales_ptr=scales, zeros_ptr=zeros, scales_a_ptr=scales_a,
        M=M, N=N, K=K,
        W_nbits=W_nbits, group_size=group_size, 
        unpack_mask=(1 << W_nbits) - 1, elements_per_sample=elements_per_sample,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        stride_meta_g=stride_meta_g, stride_meta_n=stride_meta_n,
        input_dtype=tl.float16, output_dtype=tl.float16, acc_dtype=tl.float32, meta_dtype=tl.float16,
        channel_scale_mode=1, W_group_mode=1, zero_is_scalar=0,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        A_load_order=1, meta_evict_policy='evict_last', data_contiguous=1,
    )
    
    # For verification, compute reference result using PyTorch
    # Dequantize the weights
    b_dequantized = torch.zeros((K, N), dtype=torch.float16, device='cuda')
    for g in range(K // group_size):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, K)
        for i in range(start_idx, end_idx):
            element_idx = i % elements_per_sample
            packed_idx = i // elements_per_sample
            shift = element_idx * W_nbits
            mask = (1 << W_nbits) - 1
            b_dequantized[i] = ((b_packed[packed_idx] >> shift) & mask).to(torch.float16)
            # Apply scales
            b_dequantized[i] *= scales[g]
    
    # Compute reference result
    c_ref = torch.matmul(a, b_dequantized)
    
    # Check correctness
    max_diff = torch.max(torch.abs(c_ref - c_triton))
    print(f"Max difference between PyTorch and Triton: {max_diff.item()}")
    
    # Benchmark
    warmup = 25
    rep = 100
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(warmup + rep):
        gemm_kernel[grid](
            a_ptr=a, b_ptr=b_packed, c_ptr=c_triton,
            scales_ptr=scales, zeros_ptr=zeros, scales_a_ptr=scales_a,
            M=M, N=N, K=K, 
            W_nbits=W_nbits, group_size=group_size, 
            unpack_mask=(1 << W_nbits) - 1, elements_per_sample=elements_per_sample,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cm=stride_cm, stride_cn=stride_cn,
            stride_meta_g=stride_meta_g, stride_meta_n=stride_meta_n,
            input_dtype=tl.float16, output_dtype=tl.float16, acc_dtype=tl.float32, meta_dtype=tl.float16,
            channel_scale_mode=1, W_group_mode=1, zero_is_scalar=0,
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
            A_load_order=1, meta_evict_policy='evict_last', data_contiguous=1,
        )
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed_time = (end - start) / rep
    
    print(f"Triton kernel time: {elapsed_time * 1000:.2f} ms")

    
    return c_triton, c_ref, max_diff.item()

if __name__ == "__main__":
    test_kernel()