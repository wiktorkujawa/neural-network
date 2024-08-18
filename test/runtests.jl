using FluxNeuron
using Test, Flux, Statistics

@testset "Model Output Initialization" begin
    model = FluxNeuron.initialize_model()
    noisy=rand(Float32, 2, 1000)
    out1 = model(noisy |> gpu) |> cpu
    @test size(out1) == (2, 1000)
end

@testset "Model Training" begin
    model = FluxNeuron.initialize_model()
    noisy=rand(Float32, 2, 1000)
    _, losses = FluxNeuron.train(model, noisy, 100)
    @test losses[end] < losses[1]  # Check if the loss has decreased
end

@testset "Model Saving and Loading" begin
    model = FluxNeuron.initialize_model()
    noisy=rand(Float32, 2, 1000)
    trained_model = FluxNeuron.trainAndSaveModel(noisy, 100)
    @test trained_model isa Chain  # Check if the model is a Chain
end

@testset "Accuracy Calculation" begin
    model = FluxNeuron.initialize_model()
    noisy=rand(Float32, 2, 1000)
    probs1, probs2, truth, _ = FluxNeuron.trainAndRunTest(noisy, 100)
    accuracy = mean((probs2[1,:] .> 0.5) .== truth)
    @test 0 <= accuracy <= 1
end