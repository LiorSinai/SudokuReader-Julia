#=
Machine learning utils

Lior Sinai, 1 August 2021
=#
using Random
using Images
using DataFrames, CSV
using Flux
using Flux: onecold, batch, unsqueeze

### Load data

function load_data(data_dir::AbstractString)
    # data is images within a folder with name dir/label/filename.png
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    data = Array{Float32, 3}[] # Flux expects Float32 only, else raises a warning
    labels = Int[]
    for digit in digits
        digit_dir = joinpath(data_dir, string(digit))
        println("Loading data from $digit_dir")
        for filename in readdir(digit_dir)
            image = load(joinpath(data_dir, string(digit), filename))
            image = Flux.unsqueeze(Float32.(image), 3)
            push!(data, image)
            push!(labels, digit)
        end
    end
    batch(data), labels
end

function load_mnist_data(rng::AbstractRNG, filepath::AbstractString; num_samples::Int=-1)
    # data is in CSV format
    data = Array{Float32, 3}[] # Flux expects Float32 only, else raises a warning
    labels = Int[]
    df = CSV.read(filepath, DataFrame)
    if num_samples > 0
        idxs = randperm(rng, nrow(df))[1:num_samples]
        df = df[idxs, :]
    end
    for row in eachrow(df)
        image = permutedims(reshape(collect(row[2:end]), (28, 28)))/(255f0)
        image = unsqueeze(Float32.(image), 3)
        push!(data, image)
        push!(labels, row[1])
        end
    batch(data), labels
end

load_mnist_data(filepath::AbstractString; num_samples::Int=1) = 
    load_mnist_data(rng, filepath; num_samples=num_samples)

### Split data    

function split_data(rng::AbstractRNG, X::Array, y::Vector; test_split::Float64=0.2)
    n = size(X)[end]
    n_train = n - round(Int, test_split * n)
    
    idxs = collect(1:n)
    randperm!(rng, idxs)

    inds_start = ntuple(Returns(:), ndims(data) - 1)
    X_ = X[inds_start..., idxs]
    y_ = y[idxs]

    x_train = X_[inds_start..., 1:n_train]
    y_train = y_[1:n_train]
    x_test  = X_[inds_start..., n_train+1:end]
    y_test  = y_[n_train+1:end]

    x_train, y_train, x_test, y_test
end

split_data(X, y; test_split::Float64=0.2) = split_data(Random.GLOBAL_RNG, X, y; test_split=test_split)

function count_parameters(model)
    n_params = 0
    for layer in model
        if !isempty(params(layer))
            n_params += sum(length, params(layer)) # includes non-trainable parameters
        end
    end
    n_params
end

### Training

accuracy(ŷ::AbstractMatrix, y::AbstractMatrix) = mean(onecold(ŷ) .== onecold(y))

function train!(loss, model, data, opt_state, val_data; 
    num_epochs::Int=10, output_path="model")
    history = Dict(
        "train_acc" => Float64[], 
        "train_loss" => Float64[], 
        "val_acc" => Float64[], 
        "val_loss" => Float64[]
        )
    for epoch in 1:num_epochs
        print("$epoch ")
        for Xy in data
            input, labels = Xy
            val, grads = Flux.withgradient(model) do m
                result = m(input)
                loss(result, labels)
            end
            Flux.update!(opt_state, model, grads[1])
            print('.')
        end
        println("")
        update_history!(history, model, loss, data.data, val_data.data)
        save_path = output_path * "_e$epoch" * ".bson"
        BSON.bson(save_path,  Dict(:model=>model, :history=>history))
        println("")
    end
    println("")
    history
end

function update_history!(history::Dict, model, loss, train_data::Tuple, val_data::Tuple)
    result = model(train_data[1])
    train_acc = accuracy(result, train_data[2])
    train_loss = loss(result, train_data[2])
    val_result = model(val_data[1])
    val_acc = accuracy(val_result, val_data[2])
    val_loss = loss(val_result, val_data[2])
    
    push!(history["train_acc"], train_acc)
    push!(history["train_loss"], train_loss)
    push!(history["val_acc"], val_acc)
    push!(history["val_loss"], val_loss)

    @printf "train_acc=%.4f%%; " train_acc * 100
    @printf "train_loss=%.4f; " train_loss
    @printf "val_acc=%.4f%%; " val_acc * 100
    @printf "val_loss=%.4f; " val_loss
end


### Reporting

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
    C
end
