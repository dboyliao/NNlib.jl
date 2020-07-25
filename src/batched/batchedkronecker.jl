export batched_kronecker

"""
mat1: (m, n, batch)
mat2: (j, k, batch)
"""
function batched_kronecker(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}) where {T1, T2}
    size(A, 3) == size(B, 3) || throw(DimensionMismatch("batch size mismatch"))
    num_batch = size(A, 3)
    A_flatten = reshape(A, :, 1, num_batch)
    B_flatten = reshape(B, 1, :, num_batch)
    curried_reshape(new_shape...) = (m)-> reshape(m, new_shape...)
    curried_permdims(perms...) = (m) -> permutedims(m, perms)
    batched_mul(
        A_flatten,
        B_flatten,
        ) |> 
    curried_reshape(size(A)[1:end-1]..., size(B)[1:end-1]..., num_batch) |>
    curried_permdims(3, 1, 4, 2, 5) |>
    curried_reshape(size(A, 1)*size(B, 1), size(A, 2)*size(B, 2), num_batch)
end