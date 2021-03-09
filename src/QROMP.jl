module QROMP

using LinearAlgebra
using LoopVectorization

include("qr.jl")
include("omp.jl")

end # module
