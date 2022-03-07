include("data_utils.jl")
include("../diffusion/DiffusionModels.jl")
include("denoising_model.jl")

using Flux
using Flux.Data: DataLoader
using .DiffusionModels


timesteps = 1000
device = gpu

model = TransformerDDPM() |> device

println("Model params: $(count_params(model))")
betas = collect(LinRange(1e-6, 1e-2, 1000)) |> device
diffusion = GaussianDiffusionModel(model, betas, timesteps, (32, 512), device)

train_x, eval_set = get_dataset(;limit=10000)

train_loader = DataLoader(train_x, batchsize=16, shuffle=true)

trainer = Trainer(diffusion, train_loader, 1e-3, 100)
train(trainer; save_model=true)