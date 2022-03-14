include("data_utils.jl")
include("../diffusion/DiffusionModels.jl")
include("denoising_model.jl")

using Flux
using Flux.Data: DataLoader
using .DiffusionModels


timesteps = 1000
batch_size=64
device = gpu

model = DenseDDPM(512) |> device

println("Model params: $(count_params(model))")
betas = collect(LinRange(1e-6, 1e-2, 1000)) |> device
diffusion = GaussianDiffusionModel(model, betas, timesteps, (512, 32), device)

train_x, test_x = get_dataset(;limit=10000)

train_loader = DataLoader(train_x, batchsize=batch_size, shuffle=true)
test_loader = DataLoader(test_x, batchsize=batch_size, shuffle=true)

trainer = Trainer(diffusion, train_loader, test_loader, 1e-3, 100)
train!(trainer; save_model=true)