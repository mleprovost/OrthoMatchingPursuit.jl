
@testset "Test the LS and QR versions of the OMP" begin

    ψ = randn(100, 20)
    u = randn(100)
    ctrue = ψ\u

    # Force a greedy approach over the entire set of indices
    # With LS
    idxls, cls, ϵls = lsomp(ψ, u; invert = true, verbose = true, ϵrel = eps())

    # With QR
    idxqr, cqr, ϵqr = qromp(ψ, u; invert = true, verbose = true, ϵrel = eps())

    @test norm(cls-cqr)<1e-14
    @test norm(idxls-idxqr)<1e-14
    @test norm(ctrue[idxqr]-cls)<1e-14
    @test norm(ϵls-ϵqr)<1e-14
end
