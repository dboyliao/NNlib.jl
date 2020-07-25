export batched_outer

"""
# Parameters

- `A`: array of shape (`m`, `num_batch`)
- `B`: array of shape (`n`, `num_batch`)

# Return

- `C`: array of shape (`m`, `n`, `num_batch`), where `C[:, :, i] = A[:, i] * B[:, i]'`

"""
function batched_outer(A, B)
    @assert ndims(A) == 2
    @assert ndims(B) == 2
    size(A, 2) == size(B, 2) || throw(DimensionMismatch("batch size mismatch"))
    num_batch = size(A, 2)
    A′ = reshape(A, :, 1, num_batch)
    B′ = reshape(B, 1, :, num_batch)
    batched_mul(A′, B′)
end