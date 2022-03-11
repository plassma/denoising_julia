using Flux: @functor
using Flux
using CUDA
using OMEinsum, SymEngine
include("../nn_common.jl")

struct NDBatchNorm
    batch_norm::BatchNorm
end

@functor NDBatchNorm

NDBatchNorm(dim) = NDBatchNorm(BatchNorm(dim))

function (self::NDBatchNorm)(x)
    shape = size(x)
    x = reshape(x, self.batch_norm.chs, :)
    x = self.batch_norm(x)
    reshape(x, shape)
end

struct DenseFiLM
    chain::Chain
    scale::Dense
    shift::Dense
end

@functor DenseFiLM

DenseFiLM(dim, out_dim) = DenseFiLM(Chain(SinusoidalPosEmb(dim), Dense(dim, dim * 4, swish), Dense(dim * 4, dim * 4, swish)), Dense(dim * 4, out_dim), Dense(dim * 4, out_dim))

function (df::DenseFiLM)(t)
    t = df.chain(t)
    shift, scale = df.scale(t), df.shift(t)
    shape = size(shift)
    return reshape(shift, shape[1], 1, shape[2:end]...), reshape(scale, shape[1], 1, shape[2:end]...)
end

featurewise_affine(x, scale, shift) = scale * x + shift

struct DenseResBlock
    chain::Chain
    shortcut
end

@functor DenseResBlock

DenseResBlock(in_dim::Int64, out_dim::Int64) = DenseResBlock(Chain(NDBatchNorm(in_dim), Dense(in_dim, out_dim), NDBatchNorm(out_dim), Dense(out_dim, out_dim)), in_dim == out_dim ? identity : Dense(in_dim, out_dim))

function (self::DenseResBlock)(in, scale, shift)
    x = self.chain[1](in)
    x = featurewise_affine.(x, scale, shift)
    x = swish.(x)
    x = self.chain[2:3](x)
    x = featurewise_affine.(x, scale, shift)
    x = swish.(x)
    x = self.chain[4](x)
    x + self.shortcut(in)
end

struct DenseDDPM
    chain::Chain
    num_layers::Int64
end

@functor DenseDDPM

function DenseDDPM(in_dims, num_layers=3, mlp_dims=2048, channels=32)
    layers = []
    push!(layers, Dense(in_dims, mlp_dims))

    for _ = 1:num_layers
        push!(layers, DenseFiLM(128, mlp_dims))
        push!(layers, DenseResBlock(mlp_dims, mlp_dims))
    end

    push!(layers, NDBatchNorm(mlp_dims))
    push!(layers, Dense(mlp_dims, in_dims))
    DenseDDPM(Chain(layers...), num_layers)
end

function (self::DenseDDPM)(x, t)
    x = self.chain[1](x)

    for i = 0:self.num_layers - 1
        scale, shift = self.chain[i * 2 + 2](t)
        x = self.chain[i * 2 + 3](x, scale, shift)
    end
    x = self.chain[end - 1](x)
    x = self.chain[end](x)
end

struct TransformerDDPM
    layers::Chain
    pos_emb::AbstractArray
    num_attention_layers::Int64
    num_mlp_layers::Int64
end

@functor TransformerDDPM

function TransformerDDPM(dim=512, heads=8, attention_layers=6, mlp_layers=2, mlp_dims=2048, channels=32, emb_dim=dim)
    pos_emb = reshape(SinusoidalPosEmb(emb_dim)(1:channels), emb_dim, channels, 1)
    layers = []
    push!(layers, Dense(dim, emb_dim))

    for i = 1:attention_layers
        push!(layers, NDBatchNorm(emb_dim))
        push!(layers, SelfAttention(emb_dim, heads))
        push!(layers, NDBatchNorm(emb_dim))
        push!(layers, Dense(emb_dim, mlp_dims, gelu))
        push!(layers, Dense(mlp_dims, emb_dim, gelu))
    end

    push!(layers, NDBatchNorm(emb_dim))
    push!(layers, Dense(emb_dim, mlp_dims))

    for i = 1:mlp_layers
        push!(layers, DenseFiLM(128, mlp_dims))
        push!(layers, DenseResBlock(mlp_dims, mlp_dims))
    end

    push!(layers, NDBatchNorm(mlp_dims))
    push!(layers, Dense(mlp_dims, dim))
    TransformerDDPM(Chain(layers...), pos_emb, attention_layers, mlp_layers)
end

function (self::TransformerDDPM)(x, t)
    i = 0
    x = self.layers[i+=1](x) .+ self.pos_emb

    for _ = 1:self.num_attention_layers
        shortcut = x
        x = self.layers[(i+=1):(i+=1)](x)
        x += shortcut
        shortcut = x
        x = self.layers[(i+=1):(i+=2)](x)
        x += shortcut
    end
    x = self.layers[(i+=1):(i+=1)](x)

    for _ = 1:self.num_mlp_layers
        scale, shift = self.layers[i+=1](t)
        x = self.layers[i+=1](x, scale, shift)
    end

    self.layers[end - 1:end](x)
end