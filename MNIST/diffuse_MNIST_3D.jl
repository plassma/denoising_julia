include("UNet.jl")
include("../diffusion/DiffusionModels.jl")

using MLDatasets: MNIST
using Flux
using Flux.Data: DataLoader
using .DiffusionModels
using .UNet
using Plots
using Random

pyplot()
Plots.PyPlotBackend()

function plot_samples(samples)

    a = zeros(28*4, 28*4)

    for i = 1:(size(samples)[end])
        for j = 1:28
            for k = 1:28
                a[((i - 1) ÷ 4) * 28 + k, ((i - 1) % 4) * 28 + j] = samples[j, k, 1, i]
            end
        end
    end

    p = heatmap(a, c = :greys)
    display(p)
end

timesteps = 1000
device = gpu

unet = Unet() |> device

betas = make_beta_schedule(timesteps) |> device
diffusion = GaussianDiffusionModel(unet, betas, timesteps, (28, 28, 1), device)

train_x, _ = MNIST.traindata(Float32);
train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |> device
train_loader = DataLoader(train_x, batchsize=32, shuffle=true)

trainer = Trainer(diffusion, train_loader, 1e-3, 100)
train(trainer, plot_samples)