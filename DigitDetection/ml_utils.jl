#=
Machine learning utils

Lior Sinai, 1 August 2021

- load_data
- split_data
-count_parameters

=#

using FileIO
using Random


function load_data(inpath)
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    data = Array{Float32, 3}[] # Flux expects Float32 only, else raises a warning
    labels = Int[]
    for digit in digits
        indir = joinpath(inpath, string(digit))
        println("loading data from $indir")
        for filename in readdir(indir)
            image = load(joinpath(inpath, string(digit), filename))
            image = Flux.unsqueeze(Float32.(image), 3)
            push!(data, image)
            push!(labels, digit)
        end
    end
    data, labels
end


function split_data(X, y; rng=Random.GLOBAL_RNG, test_split=0.2)
    n = length(X)
    n_train = n - round(Int, test_split * n)
    
    idxs = collect(1:n)
    randperm!(rng, idxs)

    X_ = X[idxs]
    y_ = y[idxs]

    x_train = X_[1:n_train]
    y_train = y_[1:n_train]
    x_test  = X_[n_train+1:end]
    y_test  = y_[n_train+1:end]

    x_train, y_train, x_test, y_test
end


function count_parameters(model)
    n_params = 0
    for layer in model
        if !isempty(params(layer))
            n_params += sum(length, params(layer)) # includes non-trainable parameters
        end
    end
    n_params
end