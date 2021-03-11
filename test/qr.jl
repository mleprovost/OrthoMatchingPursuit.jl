

@testset "Verify updateqrfactUnblocked! for a vector" begin

    ψ = randn(100, 20)
    ϕ = randn(100)

    F = qrfactUnblocked(ψ)
    Fold = deepcopy(F)
    G = qrfactUnblocked(hcat(ψ, ϕ))
    F = updateqrfactUnblocked!(F, ϕ)
    @test norm(F.τ-G.τ)<1e-14
    @test norm(F.factors - G.factors)<1e-14
end

@testset "Verify updateqrfactUnblocked! for a matrix" begin

    ψ = randn(100, 20)
    ϕ = randn(100, 5)

    F = qrfactUnblocked(ψ)
    Fold = deepcopy(F)
    G = qrfactUnblocked(hcat(ψ, ϕ))
    F = updateqrfactUnblocked!(F, ϕ)
    @test norm(F.τ-G.τ)<1e-14
    @test norm(F.factors - G.factors)<1e-14
end
