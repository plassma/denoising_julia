include("data_utils.jl")
include("../diffusion/DiffusionModels.jl")
include("denoising_model.jl")

using BSON: @load
using .DiffusionModels
using Flux
using Dates
using NPZ

device = gpu
num_samples = remaining = 100
batch_size = 64

@load "results/dm_transformer.bson" model
model = model |> device
samples = Array{Float32, 3}(undef, model.data_shape..., num_samples)

while remaining > 0
    if batch_size > remaining
        global batch_size = remaining
    end
    new_samples = sample(model, batch_size) |> cpu
    
    sampled = num_samples - remaining
    samples[:, :, sampled + 1:sampled + batch_size] = new_samples

    global remaining -= batch_size
    println("Generated $(sampled + batch_size)/$(num_samples)")
end

abnormalize!(samples)

path = "sampled/$(DateTime(now()))"
mkpath(path)
npzwrite("$path/samples.npz", samples)