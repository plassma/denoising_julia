include("UNet.jl")
include("../diffusion/DiffusionModels.jl")

using MLDatasets: MNIST
using Flux
using Flux.Data: DataLoader
using .DiffusionModels
using .UNet
using Random
using Images
using Plots

global epoch = 1
function save_samples(samples, save_path="")
    samples = (reshape(samples, 28, 28, size(samples)[end]) .+ 1) ./ 2

    a = zeros(28*4, 28*4)

    for i = 1:(size(samples)[end])
        for j = 1:28
            for k = 1:28
                a[((i - 1) ÷ 4) * 28 + k, ((i - 1) % 4) * 28 + j] = samples[j, k, i]
            end
        end
    end

    save("$save_path$epoch.png", colorview(Gray, a))
    global epoch += 1
end

timesteps = 1000
device = gpu

unet = Unet() |> device

betas = collect(LinRange(1e-6, 1e-2, timesteps)) |> device
diffusion = GaussianDiffusionModel(unet, betas, (28, 28, 1), device)

train_x, _ = MNIST.traindata(Float32)
test_x, _ = MNIST.testdata(Float32)
train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |> device
test_x = 2f0 * reshape(test_x, 28, 28, 1, :) .- 1f0 |> device
train_loader = DataLoader(train_x, batchsize=32, shuffle=true)
test_loader = DataLoader(test_x, batchsize=32, shuffle=true)

trainer = Trainer(diffusion, train_loader, test_loader, 1e-3, 10)
train!(trainer; handle_samples=save_samples)