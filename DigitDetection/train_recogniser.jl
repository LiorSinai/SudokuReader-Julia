#=
NUmber Recognition

Lior Sinai, 1 August 2021
=#

using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, logitcrossentropy
using BSON, JSON
using StatsBase: mean
using Random
using Printf

include("ml_utils.jl")
include("models/cnn_Flux.jl")
include("models/cnn.jl")
include("models/LeNet5.jl")
include("models/multilayer_perceptron.jl")

#### load data
char74k_filepath = "../../../datasets/Char74k/74k_numbers_28x28/"
@time data_char74k, labels_char74k = load_data(char74k_filepath);

data = data_char74k 
labels = labels_char74k
println("data sizes: ", size(data), ", ", size(labels))

### transform test set
seed = 227
rng = MersenneTwister(seed)
x_train, y_train, x_test, y_test = split_data(rng, data, labels; test_split=0.2);
y_train = onehotbatch(y_train, 0:9)
n_valid = floor(Int, 0.8 * length(y_test))
val_data = (x_test[:, :, :, 1:n_valid], onehotbatch(y_test[1:n_valid], 0:9))
test_data = (x_test[:, :, :, n_valid+1:end], onehotbatch(y_test[n_valid+1:end], 0:9))

println("train data:      ", size(x_train), ", ", size(y_train))
println("validation data: ", size(val_data[1]), ", ", size(val_data[2]))
println("test data:       ", size(test_data[1]), ", ", size(test_data[2]))

train_loader = Flux.DataLoader((x_train, y_train); batchsize=128, shuffle=true)
val_loader = Flux.DataLoader(val_data; shuffle=false)
test_loader = Flux.DataLoader(test_data; shuffle=false)

# build model
output_dir = "outputs/LeNet5"
if !isdir(output_dir)
    mkpath(output_dir)
end
output_path = joinpath(output_dir, "LeNet5")
model = LeNet5()
display(model)
println("")

# compile model
model(x_train[:, :, :, 1:10]); 

# definitions
loss(x, y) = Flux.logitcrossentropy(x, y)
loss(x::Tuple) = loss(x[1], x[2])
opt = ADAM()

@info("Beginning training loop...")
start_time = time_ns()
opt_state = Flux.setup(opt, model)
history = train!(
    loss, model, train_loader, opt_state, val_loader
    ; num_epochs=20, output_path=output_path
    )
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

test_acc = accuracy(model(test_data[1]), test_data[2])
@printf "test accuracy for %d samples: %.4f\n" size(test_data[2])[end] test_acc

history_path = joinpath(output_dir, "history.json")
open(history_path, "w") do f
    JSON.print(f, history)
end
