using Flux: @functor
using Flux
using CUDA

struct SinusoidalPosEmb #todo: move this to common
    emb::AbstractArray{Float32}
    scale::Float32
end

@functor SinusoidalPosEmb

SinusoidalPosEmb(dim::Int64, scale::Float32=1f0) = dim % 2 == 0 ? SinusoidalPosEmb(ℯ .^ ((0:(dim÷2-1)) * -(log(10000) / (dim ÷ 2 - 1))), scale) : AssertionError("dim must be even")

function (s::SinusoidalPosEmb)(x)
    emb = x' .* s.emb .* s.scale
    return cat(sin.(emb), cos.(emb), dims = 1)
end

struct DenseFiLM
    chain::Chain
    scale::Dense
    shift::Dense
end

@functor DenseFiLM

DenseFiLM(dim, out_dim) = DenseFiLM(Chain(SinusoidalPosEmb(dim, 5000f0), Dense(dim, dim * 4, swish), Dense(dim * 4, dim * 4, swish)), Dense(dim * 4, out_dim), Dense(dim * 4, out_dim))

function (df::DenseFiLM)(t)
    t = df.chain(t)
    shift, scale = df.scale(t), df.shift(t)
    return reshape(shift, 1, size(shift)...), reshape(scale, 1, size(scale)...)
end

featurewise_affine(x, scale, shift) = scale * x + shift

struct DenseResBlock
    chain::Chain
    shortcut
end

@functor DenseResBlock

DenseResBlock(in_dim::Int64, out_dim::Int64, channels::Int64) = DenseResBlock(Chain(NDBatchNorm(in_dim * channels), Dense2D(in_dim, out_dim), NDBatchNorm(out_dim * channels), Dense2D(out_dim, out_dim)), in_dim == out_dim ? identity : Dense2D(in_dim, out_dim))

function (self::DenseResBlock)(in, scale, shift)
    x = self.chain[1](in)
    x = featurewise_affine.(x, scale, shift)
    x = swish.(x)
    x = self.chain[2:3](x)
    x = featurewise_affine.(x, scale, shift)
    x = swish.(x)
    x = self.chain[4](x)
    return x + self.shortcut(in)
end

struct Dense2D
    dense::Dense
end

@functor Dense2D

Dense2D(in, out, σ=identity) = Dense2D(Dense(in, out, σ))

function (self::Dense2D)(x)
    d1, d2, B = size(x)#32, 512, batch_size
    x = reshape(x, d2, d1 * B)
    result = self.dense(x)
    return reshape(result, d1, :, B)
end

struct NDBatchNorm
    batch_norm::BatchNorm
end

@functor NDBatchNorm

NDBatchNorm(dim) = NDBatchNorm(BatchNorm(dim))

function (self::NDBatchNorm)(x)
    shape = size(x)
    x = reshape(x, :, shape[end])
    x = self.batch_norm(x)
    return reshape(x, shape)
end

struct DenseDDPM
    chain::Chain
    num_layers::Int64
end

@functor DenseDDPM

function DenseDDPM(in_dims, num_layers=3, mlp_dims=2048, channels=32)
    layers = []
    push!(layers, Dense2D(in_dims, mlp_dims))

    for _ = 1:num_layers
        push!(layers, DenseFiLM(128, mlp_dims))
        push!(layers, DenseResBlock(mlp_dims, mlp_dims, channels))
    end

    push!(layers, NDBatchNorm(mlp_dims * channels))
    push!(layers, Dense2D(mlp_dims, in_dims))
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
    return x
end