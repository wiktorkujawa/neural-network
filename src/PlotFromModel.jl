using Plots  # to draw the above figure
using FluxNeuron, CUDA, Flux, JLD2, Statistics

println("Number of steps(default = 1000): ")

#Take single line user input from the user
try
    global steps = parse(Int, readline())
catch
    global steps = 1000
end

noisy=rand(Float32, 2, 1000)
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}
# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2)) |> gpu        # move model to GPU, if available


optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
# The model encapsulates parameters, randomly initialised. Its initial output is:
out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}
probs1 = softmax(out1)      # normalise to get probabilities
BSON.@load "models/model-$(steps).bson" model

model = model |> gpu 

optim # parameters, momenta and output have all changed
out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)
probs2 = softmax(out2)      # normalise to get probabilities
mean((probs2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!

p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network($(steps) x)", legend=false)

plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))

