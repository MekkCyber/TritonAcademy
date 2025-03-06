import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
import pdb

@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  # Strides for matrix A (M, K)
        stride_bk, stride_bn,  # Strides for matrix B (K, N)
        stride_cm, stride_cn,  # Strides for matrix C (M, N)
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  # Tile sizes
        GROUP_SIZE_M: tl.constexpr,  # Number of M-dimension tiles per group (for L2 cache optimization)
        ACTIVATION: tl.constexpr  # Optional activation function to apply
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    
    Example:
    For M=16, N=8, K=16 with BLOCK_SIZE_M=2, BLOCK_SIZE_N=2, BLOCK_SIZE_K=2, GROUP_SIZE_M=3:
    
    Matrix A (16x16):
    [A00, A01, A02, ..., A0,15]
    [A10, A11, A12, ..., A1,15]
    [...                     ]
    [A15,0, A15,1, ..., A15,15]
    
    Matrix B (16x8):
    [B00, B01, B02, ..., B07]
    [B10, B11, B12, ..., B17]
    [...                   ]
    [B15,0, B15,1, ..., B15,7]
    
    Matrix C (16x8):
    [C00, C01, C02, ..., C07]
    [C10, C11, C12, ..., C17]
    [...                   ]
    [C15,0, C15,1, ..., C15,7]
    
    - We divide matrices into blocks of size 2x2
    - Matrix A (16x16) has 8x8 blocks
    - Matrix B (16x8) has 8x4 blocks
    - Matrix C (16x8) has 8x4 blocks
    - We'll have 32 thread blocks computing the 32 blocks of C
    """
    # -----------------------------------------------------------
    # STEP 1: Determine which block of the output matrix C this thread block will compute
    # -----------------------------------------------------------
    # Each thread block is assigned a unique program ID (pid)
    pid = tl.program_id(axis=0)
    
    # Calculate how many blocks we need in each dimension
    # For our example: num_pid_m = ceil(16/2) = 8, num_pid_n = ceil(8/2) = 4
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # Number of blocks in M dimension
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # Number of blocks in N dimension

    # L2 cache optimization: Group blocks along M dimension to promote data reuse
    # For our example: num_pid_in_group = 3*4 = 12 (total blocks in a group)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Determine which group this thread block belongs to
    # We have 3 groups, theoretically they all have 12 blocks
    # For pid<12: group_id = 0
    # For 12<=pid<24: group_id = 1
    # else: group_id = 2
    group_id = pid // num_pid_in_group
    
    # Find the first block index in M dimension for this group
    # For group_id=0: first_pid_m = 0
    # For group_id=1: first_pid_m = 3
    # For group_id=2: first_pid_m = 6
    first_pid_m = group_id * GROUP_SIZE_M
    
    # Calculate actual group size (might be smaller at boundaries)
    # For first_pid_m=0: group_size_m = min(8-0, 3) = 3
    # For first_pid_m=3: group_size_m = min(8-3, 3) = 3
    # For first_pid_m=6: group_size_m = min(8-6, 3) = 2 (last group is smaller)
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Calculate the specific block indices this thread block will compute
    # For pid=0: pid_m = 0 + ((0 % 12) % 3) = 0, pid_n = (0 % 12) // 3 = 0
    # For pid=1: pid_m = 0 + ((1 % 12) % 3) = 1, pid_n = (1 % 12) // 3 = 0
    # For pid=2: pid_m = 0 + ((2 % 12) % 3) = 2, pid_n = (2 % 12) // 3 = 0
    # For pid=3: pid_m = 0 + ((3 % 12) % 3) = 0, pid_n = (3 % 12) // 3 = 1
    # For pid=4: pid_m = 0 + ((4 % 12) % 3) = 1, pid_n = (4 % 12) // 3 = 1
    # For pid=5: pid_m = 0 + ((5 % 12) % 3) = 2, pid_n = (5 % 12) // 3 = 1
    # For pid=6: pid_m = 0 + ((6 % 12) % 3) = 0, pid_n = (6 % 12) // 3 = 2
    # For pid=7: pid_m = 0 + ((7 % 12) % 3) = 1, pid_n = (7 % 12) // 3 = 2
    # For pid=8: pid_m = 0 + ((8 % 12) % 3) = 2, pid_n = (8 % 12) // 3 = 2
    # For pid=9: pid_m = 0 + ((9 % 12) % 3) = 0, pid_n = (9 % 12) // 3 = 3
    # For pid=10: pid_m = 0 + ((10 % 12) % 3) = 1, pid_n = (10 % 12) // 3 = 3
    # For pid=11: pid_m = 0 + ((11 % 12) % 3) = 2, pid_n = (11 % 12) // 3 = 3
    # For pid=12: pid_m = 3 + ((12 % 12) % 3) = 3, pid_n = (12 % 12) // 3 = 0
    # For pid=13: pid_m = 3 + ((13 % 12) % 3) = 4, pid_n = (13 % 12) // 3 = 0
    # For pid=14: pid_m = 3 + ((14 % 12) % 3) = 5, pid_n = (14 % 12) // 3 = 0
    # For pid=15: pid_m = 3 + ((15 % 12) % 3) = 3, pid_n = (15 % 12) // 3 = 1
    # For pid=16: pid_m = 3 + ((16 % 12) % 3) = 4, pid_n = (16 % 12) // 3 = 1
    # For pid=17: pid_m = 3 + ((17 % 12) % 3) = 5, pid_n = (17 % 12) // 3 = 1
    # For pid=18: pid_m = 3 + ((18 % 12) % 3) = 3, pid_n = (18 % 12) // 3 = 2
    # For pid=19: pid_m = 3 + ((19 % 12) % 3) = 4, pid_n = (19 % 12) // 3 = 2
    # For pid=20: pid_m = 3 + ((20 % 12) % 3) = 5, pid_n = (20 % 12) // 3 = 2
    # For pid=21: pid_m = 3 + ((21 % 12) % 3) = 3, pid_n = (21 % 12) // 3 = 3
    # For pid=22: pid_m = 3 + ((22 % 12) % 3) = 4, pid_n = (22 % 12) // 3 = 3
    # For pid=23: pid_m = 3 + ((23 % 12) % 3) = 5, pid_n = (23 % 12) // 3 = 3
    # For pid=24: pid_m = 6 + ((24 % 12) % 3) = 6, pid_n = (24 % 12) // 3 = 0
    # For pid=25: pid_m = 6 + ((25 % 12) % 3) = 7, pid_n = (25 % 12) // 3 = 0
    # For pid=26: pid_m = 6 + ((26 % 12) % 3) = 6, pid_n = (26 % 12) // 3 = 1
    # For pid=27: pid_m = 6 + ((27 % 12) % 3) = 7, pid_n = (27 % 12) // 3 = 1
    # For pid=28: pid_m = 6 + ((28 % 12) % 3) = 6, pid_n = (28 % 12) // 3 = 2
    # For pid=29: pid_m = 6 + ((29 % 12) % 3) = 7, pid_n = (29 % 12) // 3 = 2
    # For pid=30: pid_m = 6 + ((30 % 12) % 3) = 6, pid_n = (30 % 12) // 3 = 3
    # For pid=31: pid_m = 6 + ((31 % 12) % 3) = 7, pid_n = (31 % 12) // 3 = 3
    #
    # Matrix C (16x8) with blocks of size 2x2 will be computed by these pids:
    # [pid=0,  pid=3,  pid=6,  pid=9 ]
    # [pid=1,  pid=4,  pid=7,  pid=10]
    # [pid=2,  pid=5,  pid=8,  pid=11]
    # [pid=12, pid=15, pid=18, pid=21]
    # [pid=13, pid=16, pid=19, pid=22]
    # [pid=14, pid=17, pid=20, pid=23]
    # [pid=24, pid=26, pid=28, pid=30]
    # [pid=25, pid=27, pid=29, pid=31]
    #
    # Swizzle pattern visualization:
    # Group 0:                Group 1:                Group 2:
    # +---+---+---+---+      +---+---+---+---+      +---+---+---+---+
    # | 0 | 3 | 6 | 9 |      |12 |15 |18 |21 |      |24 |26 |28 |30 |
    # +---+---+---+---+      +---+---+---+---+      +---+---+---+---+
    # | 1 | 4 | 7 |10 |      |13 |16 |19 |22 |      |25 |27 |29 |31 |
    # +---+---+---+---+      +---+---+---+---+      +---+---+---+---+
    # | 2 | 5 | 8 |11 |      |14 |17 |20 |23 |      +---+---+---+---+
    # +---+---+---+---+      +---+---+---+---+
    #
    # Notice how threads are assigned in column-major order within each group:
    # - Within each group, we process blocks in a column-first pattern (0,1,2 then 3,4,5 etc.)
    # - This creates spatial locality for memory accesses within each group
    # - Adjacent thread blocks process adjacent memory, improving cache efficiency
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # -----------------------------------------------------------
    # STEP 2: Create pointers to the blocks of input matrices A and B
    # -----------------------------------------------------------
    # Calculate offsets for the block of A and B this thread block will process
    # For pid=13 (computing block in row 4, column 0 of C):
    # offs_am = [8,9] (rows 8-9 of A)
    # offs_bn = [0,1] (columns 0-1 of B)
    # offs_k = [0,1] (columns 0-1 of A / rows 0-1 of B)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointer blocks for A and B
    # Strides represent the memory distance between consecutive elements in a dimension:
    # - stride_am: bytes between consecutive rows in matrix A (used to move between rows of A)
    # - stride_ak: bytes between consecutive columns in matrix A (used to move between columns of A)
    # - stride_bk: bytes between consecutive rows in matrix B (used to move between rows of B)
    # - stride_bn: bytes between consecutive columns in matrix B (used to move between columns of B)
    #
    # For pid=13 (computing block in row 4, column 0 of C):
    # a_ptrs calculates pointers to A[8:10, 0:2] by:
    #   - Starting at base pointer a_ptr
    #   - Adding row offsets (offs_am[:, None] * stride_am) to move to rows 8-9
    #   - Adding column offsets (offs_k[None, :] * stride_ak) to access columns 0-1
    # b_ptrs calculates pointers to B[0:2, 0:2] by:
    #   - Starting at base pointer b_ptr
    #   - Adding row offsets (offs_k[:, None] * stride_bk) to move to rows 0-1
    #   - Adding column offsets (offs_bn[None, :] * stride_bn) to access columns 0-1
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # -----------------------------------------------------------
    # STEP 3: Compute the matrix multiplication C = A × B block by block
    # -----------------------------------------------------------
    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # For our example with K=16 and BLOCK_SIZE_K=2, we need 8 iterations
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # For pid=0, iteration 0: we load A[0:2, 0:2] and B[0:2, 0:2]
        # For pid=0, iteration 1: we load A[0:2, 2:4] and B[2:4, 0:2]
        # For pid=0, iteration 2: we load A[0:2, 4:6] and B[4:6, 0:2]
        # And so on...

        # Calculate how many elements remain in the K dimension for the current iteration
        k_remaining = K - k * BLOCK_SIZE_K
        
        # a_mask handles the columns of matrix A (K dimension)
        # For example, if K=10 and BLOCK_SIZE_K=4, in the last iteration (k=2):
        # - k_remaining = 10 - 2*4 = 2
        # - offs_k = [0,1,2,3]
        # - a_mask will be [[True,True,False,False]]
        # This ensures we only load the valid 2 remaining columns
        a_mask = (offs_k[None, :] < k_remaining)
        
        # b_mask handles the rows of matrix B (K dimension)
        # Using the same example, b_mask will be:
        # [[True], [True], [False], [False]]
        # This ensures we only load the valid 2 remaining rows
        b_mask = (offs_k[:, None] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute matrix multiplication for this block and accumulate
        # For pid=0, iteration 0: 
        # C[0,0] += A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] += A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] += A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] += A[1,0]*B[0,1] + A[1,1]*B[1,1]
        accumulator = tl.dot(a, b, accumulator)
        
        # Move pointers to the next K block
        # For iteration 1: a_ptrs now points to A[0:2, 2:4]
        # For iteration 1: b_ptrs now points to B[2:4, 0:2]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # -----------------------------------------------------------
    # STEP 4: Apply activation function (if specified) and prepare for output
    # -----------------------------------------------------------
    # Apply activation function if specified
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    
    # Convert back to float16 for output
    c = accumulator.to(tl.float16)
    
    # -----------------------------------------------------------
    # STEP 5: Write the computed block back to matrix C
    # -----------------------------------------------------------
    # Calculate global indices for this block in matrix C
    # same as the offs_am and offs_bn
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointers to the output locations in C
    # same as the a_ptrs and b_ptrs
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create mask to handle boundary conditions
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Store the computed values to matrix C
    # For pid=0: This writes to C[0:2, 0:2]
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=None):
    M, K = a.shape
    K, N = b.shape
    # Initialize output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    # Calculate grid dimensions based on block sizes
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch the kernel
    matmul_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=c.data_ptr(),
        M=M, N=N, K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        ACTIVATION=activation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c

# Test function for the matmul implementation
def test_matmul():
    import torch    
    for m, n, k in [(32, 32, 32), (256, 512, 128), (1024, 1024, 1024)]:
        a = torch.randn((m, k), device='cuda', dtype=torch.float16)
        b = torch.randn((k, n), device='cuda', dtype=torch.float16)
        c_ref = torch.matmul(a, b)
        c_triton = matmul(a, b)
        assert torch.allclose(c_ref, c_triton, rtol=1e-2, atol=1e-2), f"Failed for size {m}x{k}x{n}"
    
    print("Size tests passed!")

if __name__ == "__main__":
    test_matmul()
