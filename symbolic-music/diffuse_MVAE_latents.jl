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
device = gpu


model = DenseDDPM(512) |> device

println("Model params: $(count_params(model))")
betas = make_beta_schedule(timesteps) |> device
diffusion = GaussianDiffusionModel(model, betas, timesteps, (32, 512), device)

train_x, eval_set = get_dataset(;limit=10000)

train_loader = DataLoader(train_x, batchsize=64, shuffle=true)

trainer = Trainer(diffusion, train_loader, 1e-3, 100)
train(trainer; save_model=false)