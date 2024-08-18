using Plots  # to draw the above figure
using FluxNeuron

println("Number of steps(default = 1000): ")

#Take single line user input from the user
try
    global steps = parse(Int, readline())
catch
    global steps = 1000
end

noisy=rand(Float32, 2, 1000)
probs1, probs2, truth, losses = trainAndRunTest(noisy, steps)

p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network($(steps) x)", legend=false)

plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))