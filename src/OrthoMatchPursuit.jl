
__precompile__(true)

module OrthoMatchPursuit

using LinearAlgebra
using ElasticArrays

include("qr.jl")
include("elasticqr.jl")
include("lsomp.jl")
include("pivotedqr.jl")
include("qromp.jl")
include("batchqromp.jl")
include("wrapper.jl")

end # module
