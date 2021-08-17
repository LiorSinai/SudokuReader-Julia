#=
Machine learning utils

Lior Sinai, 1 August 2021

- load_data
- split_data
-count_parameters

=#

using FileIO
using Random
using DataFrames, CSV


function load_data(inpath)
    # data is images within a folder with name dir/label/filename.png
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


function load_mnist_data(inpath)
    # data is in CSV format
    data = Array{Float32, 3}[] # Flux expects Float32 only, else raises a warning
    labels = Int[]
    df = CSV.read(inpath, DataFrame)
    for row in eachrow(df)
        image = permutedims(reshape(collect(row[2:end]), (28, 28)))/(255f0)
        image = Flux.unsqueeze(Float32.(image), 3)
        push!(data, image)
        push!(labels, row[1])
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


"""
confusion_matrix(ŷ, y) \right Matrix{T}

Returns a confusion matrix. Rows are the actual classes and the columns are the predicted classes
"""
function confusion_matrix(ŷ::Vector{T}, y::Vector{T}, labels=1:maximum(y)) where T
    if length(y) != length(ŷ)
        throw(DimensionMismatch("y_actual length is not the same as y_pred length"))
    end
    n = length(labels)
    C = zeros(Int, n, n)
    for (i, label_actual) in enumerate(labels)
        idxs_true = y .== label_actual
        for (j, label_pred) in enumerate(labels)
            C[i, j] = count(ŷ[idxs_true] .== label_pred)
        end
    end
    return C
end
