#=
Julia Flux Convolutional Neural Network Explained
By Nigel Adams

Convolutional Neural Network MNIST Example Explained
By Clark Fitzgerald

Apparently an old model from the Flux model zoo.

layers:     8
parameters: 16,938
size:       67.2 KiB
source:     https://spcman.github.io/getting-to-know-julia/deep-learning/vision/flux-cnn-zoo/
            http://webpages.csus.edu/fitzgerald/julia-convolutional-neural-network-MNIST-explained/

=#
using Flux

function build_model_Flux()
    return Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(288, 10),
    )
end