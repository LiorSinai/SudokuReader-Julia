#=
NUmber Recognition

Lior Sinai, 1 August 2021
=#

using Flux
using Flux: Data.DataLoader, unsqueeze
using Flux: onehotbatch, onecold, logitcrossentropy
using BSON # for saving models

using StatsBase: mean
using Random

using Printf

include("ml_utils.jl")
include("cnn_Flux.jl")
include("LeNet5.jl")
include("cnn.jl")
include("nn.jl")


#### load data
@time data_char74k, labels_char74k = load_data("../../datasets/74k_numbers_28x28/");

data = data_char74k 
labels = labels_char74k
println("data loaded\n")

### transform test set
seed = 227
rng = MersenneTwister(seed)
x_train, y_train, x_test, y_test = split_data(data, labels, rng=rng, test_split=0.2);
y_train = onehotbatch(y_train, 0:9)
y_test =  onehotbatch(y_test, 0:9)
train_data = Flux.DataLoader((Flux.batch(x_train), y_train), batchsize=128)

n_valid = floor(Int, 0.8*size(y_test, 2))
valid_data = (Flux.batch(x_test[1:n_valid]), y_test[:, 1:n_valid])
test_data = (Flux.batch(x_test[n_valid+1:end]), y_test[:, n_valid+1:end])

# build model
output_path = joinpath("DigitDetection\\models", "LeNet5")
model = LeNet5()
display(model)
println("")

# compile model
model(Flux.batch(x_train[1:10])); 

# definitions
accuracy(ŷ, y) = mean(onecold(ŷ, 0:9) .== onecold(y, 0:9))

loss(x::Tuple) = Flux.logitcrossentropy(model(x[1]), x[2])
loss(x, y) = Flux.logitcrossentropy(model(x), y)

opt=ADAM()

@info("Beginning training loop...")

# custom training loop edited from Flux.jl/src/optimise/train.jl
function train!(loss, ps, train_data, opt, acc, valid_data; n_epochs=100)
    history = Dict("train_acc"=>Float64[], "valid_acc"=>Float64[])
    for e in 1:n_epochs
        print("$e ")
        ps = Flux.Params(ps)
        for batch_ in train_data
            gs = gradient(ps) do
                loss(batch_...)
            end
            Flux.update!(opt, ps, gs)
            print('.')
        end
        # update history
        train_acc = 0.0
        n_samples = 0
        for batch_ in train_data
            train_acc += sum(onecold(model(batch_[1])) .== onecold(batch_[2]))
            n_samples += size(batch_[1], 4)
        end
        train_acc = train_acc/n_samples
        valid_acc = acc(model(valid_data[1]), valid_data[2])
        push!(history["train_acc"], train_acc)
        push!(history["valid_acc"], valid_acc)

        @printf "\ntrain_acc=%.4f valid_acc=%.4f\n" train_acc*100 valid_acc*100

        # save model
        save_path = output_path * "_e$e" * ".bson"
        BSON.@save save_path model history
    end
    history
end
start_time = time_ns()
history = train!(
    loss, params(model), train_data, opt, 
    accuracy, valid_data, n_epochs=20
    )
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

test_acc = accuracy(model(test_data[1]), test_data[2])
@printf "test accuracy for %d samples: %.4f\n" size(test_data[2], 2) test_acc

# plot history 
using Plots

canvas = plot(
    title="training",
    xlabel="epochs",
    ylabel="accuracy",
    legend=:best
    )
epochs = 1:length(history["train_acc"])
plot!(canvas, epochs, history["train_acc"], label="train")
plot!(canvas, epochs, history["valid_acc"], label="valid")
plot!(canvas, [epochs[end]], [test_acc], markershape=:star, label="test")
plot!(canvas, legend=:topleft)
plot!(canvas, ylims=(ylims(canvas)[1], 1))

savefig(canvas, "images/outputs/history.png")
canvas
