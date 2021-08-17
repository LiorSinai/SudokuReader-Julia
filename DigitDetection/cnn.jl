#=
Small models from my own experiments
=#

using Flux

function calc_output_size(input_size::Int, filter_size::Int, stride::Int=1, pad::Int=0)
    floor(Int, (input_size + 2pad - filter_size)/stride) + 1
end


"""
CNN model with Dropout

layers:     7
parameters: 18,378 (default) 5,142 (n1=6, n2=16)
size:       72.9 KiB (default), 21.195 KiB (n1=6, n2=16)
"""
function build_model_cnn_dropout(; k1=5, n1=16, k2=5, n2=32, pad=0)
    input_dim = 28
    stride = 1
    max_pool_window = 2

    output_dim = input_dim
    output_dim = calc_output_size(output_dim, k1, stride, pad)
    output_dim = calc_output_size(output_dim, max_pool_window, max_pool_window, 0)
    output_dim = calc_output_size(output_dim, k2, stride, pad)
    output_dim = calc_output_size(output_dim, max_pool_window, max_pool_window, 0)
    output_size = (output_dim, output_dim, n2)

    return Chain(
        Conv((k1, k1), 1=>n1, stride=stride, pad=pad, relu),
        MaxPool((max_pool_window, max_pool_window)),

        Conv((k2, k2), n1=>n2, stride=stride, pad=pad, relu),
        MaxPool((max_pool_window, max_pool_window)),

        Flux.flatten,
        Dropout(0.5),
        Dense(prod(output_size), 10),
    )
end


function build_model_tiny()
    input_dim = 28
    stride_ = 1
    pad = 0

    output_dim = input_dim
    output_dim = calc_output_size(output_dim, 5, stride_, pad)
    output_dim = calc_output_size(output_dim, 4, 4, 0)
    output_size = (output_dim, output_dim, 6)

    return Chain(
        Conv((5, 5), 1=>6, stride=stride_, pad=pad, relu),
        MaxPool((4, 4)),
        Flux.flatten,
        Dense(prod(output_size), 10),
    )
end


"""
Huge CNN, from a popular Medium blog post

layers:     7
parameters: 1,199,882
size:       4.56 MB
"""
function build_model_huge()
    input_dim = 28
    stride_ = 1
    pad = 0

    output_dim = input_dim
    output_dim = calc_output_size(output_dim, 3, stride_, pad)
    output_dim = calc_output_size(output_dim, 3, stride_, pad)
    output_dim = calc_output_size(output_dim, 2, 2, 0)
    output_size = (output_dim, output_dim, 64)

    return Chain(
        Conv((3, 3), 1=>32, stride=stride_, pad=pad, relu),
        Conv((3, 3), 32=>64, stride=stride_, pad=pad, relu),
        MaxPool((2, 2)),
        Dropout(0.25),
        Flux.flatten,
        Dense(prod(output_size), 128),
        Dropout(0.5),
        Dense(128, 10),
    )
end

"""
Machine learning mastery

layers:     5
parameters: 542,230
size:       2.0 MB
source:     https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""
function build_model_mastery()
    input_dim = 28
    stride_ = 1
    pad = 0

    output_dim = input_dim
    output_dim = calc_output_size(output_dim, 3, stride_, pad)
    output_dim = calc_output_size(output_dim, 2, 2, 0)
    output_size = (output_dim, output_dim, 32)

    return Chain(
        Conv((3, 3), 1=>32, stride=stride_, pad=pad, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(prod(output_size), 100),
        Dense(100, 10),
    )
end