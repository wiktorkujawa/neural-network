module FluxNeuron

# This will prompt if neccessary to install everything, including CUDA:
using Flux, CUDA, Statistics, ProgressMeter

    function initialize_model()
        model = Chain(
            Dense(2 => 3, tanh),   # activation function inside layer
            BatchNorm(3),
            Dense(3 => 2)) |> gpu        # move model to GPU, if available
        return model
    end

    function train(model, noisy, steps=1_000)
        # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:                             # 2×1000 Matrix{Float32}
        truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}
        # To train the model, we use batches of 64 samples, and one-hot encoding:
        target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
        loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);
        # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

        optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

        # Training loop, using the whole data set "steps" times:
        losses = []
        @showprogress for epoch in 1:steps
            for (x, y) in loader
                loss, grads = Flux.withgradient(model) do m
                    # Evaluate model and loss inside gradient context:
                    y_hat = m(x)
                    Flux.logitcrossentropy(y_hat, y)
                end
                Flux.update!(optim, model, grads[1])
                push!(losses, loss)  # logging, outside gradient context
            end
        end

        # If we don't want to calculate losses, we can use this:
        # @showprogress for epoch in 1:steps
        #     Flux.train!(model, loader, optim) do m, x, y
        #         y_hat = m(x)
        #         Flux.logitcrossentropy(y_hat, y)
        #     end
        # end

        return truth, losses
    end

    function trainAndRunTest(noisy=rand(Float32, 2, 1000), steps=1_000)
        
        # Define our model, a multi-layer perceptron with one hidden layer of size 3:
        model = initialize_model() # initialize model

        # The model encapsulates parameters, randomly initialised. Its initial output is:
        out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}
        probs1 = softmax(out1)      # normalise to get probabilities

        truth, losses = train(model, noisy, steps)

        # optim # parameters, momenta and output have all changed
        out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)
        probs2 = softmax(out2)      # normalise to get probabilities
        mean((probs2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!

        return probs1, probs2, truth, losses
    end
    export trainAndRunTest

    function trainAndSaveModel(noisy=rand(Float32, 2, 1000), steps=1_000)
        model = initialize_model() # initialize model

        train(model, noisy, steps)
        return model
    end
    export trainAndSaveModel


end # module FluxNeuron
