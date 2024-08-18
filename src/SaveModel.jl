using JLD2, FluxNeuron, Flux, BSON
println("Number of steps(default = 1000): ")

try
    global steps = parse(Int, readline())
catch
    global steps = 1000
end

noisy=rand(Float32, 2, 1000)

model = trainAndSaveModel(noisy, steps)

BSON.@save "models/model-$(steps).bson" model=cpu(model)

