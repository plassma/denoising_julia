include("data_utils.jl")
include("../diffusion/DiffusionModels.jl")
include("denoising_model.jl")

using MLDatasets: MNIST
using Flux
using Flux.Data: DataLoader
using .DiffusionModels
using Plots
using Random


timesteps = 1000
device = cpu

reorder(x) = reshape(x, (32, 512, :))

struct DumbDiffusionConfusor
    model
    time_emb
end


function (m::DumbDiffusionConfusor)(x, t)
    t = m.time_emb(t)

    return m.model(x + t)
end

model = DumbDiffusionConfusor(Chain(flatten, Dense(512 * 32, 512 * 32, relu), Dense(512 * 32, 512 * 32), reorder),
                    Chain(SinusoidalPosEmb(128), Dense(128, 512 * 32, gelu), reorder) )|> device

x = randn(Float32, 32, 512, 1)

y = model(x, [0])


betas = make_beta_schedule(timesteps) |> device
diffusion = GaussianDiffusionModel(model, betas, timesteps, (32, 512), device)

train_x, eval_set = get_dataset(;limit=1000)

train_loader = DataLoader(train_x, batchsize=32, shuffle=true)

trainer = Trainer(diffusion, train_loader, 1e-3, 100)
train(trainer; save_model=false)