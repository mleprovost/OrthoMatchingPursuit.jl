

@testset "Verify updateqrfactUnblocked!" begin

    ψ = randn(100, 20)
    ϕ = randn(100)

    F = qrfactUnblocked(ψ)
    Fold = deepcopy(F)
    G = qrfactUnblocked(hcat(ψ, ϕ))
    F = updateqrfactUnblocked!(F, ϕ)
    @test norm(F.τ-G.τ)<1e-14
    @test norm(F.factors - G.factors)<1e-14
end
