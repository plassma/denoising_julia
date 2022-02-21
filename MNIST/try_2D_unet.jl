include("UNet.jl")
include("../utils.jl")

using .UNet
using Flux

unet = Unet() |> gpu

x = rand(Float32, 28, 28, 1, 1) |> gpu

t = unet.time_mlp[1]([0, 0, 0]|> gpu)

y = unet(x, [0] |> gpu)

dbg_dump_var(x, "x")
dbg_dump_var(y, "y")