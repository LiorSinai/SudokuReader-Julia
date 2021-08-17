#=
Simple Julia MNIST Model

layers:     4
parameters: 25,450
size:       99.7 kiB
source:     https://fluxml.ai/Flux.jl/v0.2/examples/logreg.html

=#

using Flux

sigmoid = σ

function build_model_nn()
    return Chain(
        Flux.flatten, 
        Dense(784, 32, sigmoid),
        Dense(32, 10),
    )
end