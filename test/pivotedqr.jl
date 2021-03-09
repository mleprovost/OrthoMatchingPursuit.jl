

@testset "Test the pivoted QR" begin

    ψ = randn(100, 20)
    u = randn(100)
    m, n = size(ψ)

    ctrue = ψ\u

    # Force a greedy approach over the entire set of indices
    # With pivoted QR
    idxpivot, cpivot, ϵpivot = pivotedqr(ψ, u; invert = true, verbose = true, ϵrel = eps())

    @test norm(ctrue[idxpivot]-cpivot)<1e-14

    # Residual error must decrease
    @test all(ϵpivot[2:end]-ϵpivot[1:end-1] .< 0) == true

    for k=1:n
        cverif = view(ψ,:, idxpivot[1:k])\u
        @test abs(ϵpivot[k+1] - norm(u - view(ψ,:, idxpivot[1:k])*cverif))<1e-14
    end
end
