include("UNet.jl")
include("../utils.jl")

using .UNet
using Flux
using BSON: @save

unet = Fake1DUnet(28) |> gpu

x = rand(Float32, 28*28, 1) |> gpu
y = unet(x, [0] |> gpu)

dbg_dump_var(x, "x")
dbg_dump_var(y, "y")

@save "unet.bson" unet

println("saved")
