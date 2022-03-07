using Flux: @functor
using Flux
using CUDA
using OMEinsum, SymEngine
include("../utils.jl") #todo: remove again dbg

struct SinusoidalPosEmb #todo: move this to common
    emb::AbstractArray{Float32}
    scale::Float32
end

@functor SinusoidalPosEmb

SinusoidalPosEmb(dim::Int64, scale::Float32=1f0) = dim % 2 == 0 ? SinusoidalPosEmb(ℯ .^ ((0:(dim÷2-1)) * -(log(10000) / (dim ÷ 2 - 1))), scale) : AssertionError("dim must be even")

function (s::SinusoidalPosEmb)(x)
    emb = x' .* s.emb .* s.scale
    result = cat(sin.(emb), cos.(emb), dims = 1)
    return result
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
    return reshape(shift, 1, size(shift)...), reshape(scale, 1, size(scale)...)
end

featurewise_affine(x, scale, shift) = scale * x + shift

struct DenseResBlock
    chain::Chain
    shortcut
end

@functor DenseResBlock

DenseResBlock(in_dim::Int64, out_dim::Int64, channels::Int64) = DenseResBlock(Chain(NDBatchNorm(in_dim), Dense2D(in_dim, out_dim), NDBatchNorm(out_dim), Dense2D(out_dim, out_dim)), in_dim == out_dim ? identity : Dense2D(in_dim, out_dim))

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

Dense2D(in, out, σ=identity; bias=true) = Dense2D(Dense(in, out, σ; bias = bias))

function (self::Dense2D)(x)
    d1, d2, B = size(x)#32, 512, batch_size
    x = permutedims(x, [2, 1, 3])#todo: needed?
    x = reshape(x, d2, d1 * B)
    result = self.dense(x)
    result = reshape(result, :, d1, B)#todo: is this reshaping needed?
    return permutedims(result, [2, 1, 3])
end

#function (self::Dense2D)(x)
#    d1, d2, B = size(x) #32, 512, batch_size
#    x = reshape(x, d2, d1 * B)
#    result = self.dense(x)
#    return reshape(result, d1, :, B)
#end

struct NDBatchNorm
    batch_norm::BatchNorm
end

@functor NDBatchNorm

NDBatchNorm(dim) = NDBatchNorm(BatchNorm(dim))

function (self::NDBatchNorm)(x)
    shape = size(x)
    x = reshape(x, self.batch_norm.chs, :)
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

    push!(layers, NDBatchNorm(mlp_dims))
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

struct SelfAttention
    scale::Float32
    heads::Int64
    qkv::Tuple
    out::Dense2D
end

@functor SelfAttention

function SelfAttention(dim, heads=8, dim_head=dim ÷ heads)
    scale = dim_head ^ -0.5
    hidden_dim = dim_head * heads
    SelfAttention(scale, heads, Tuple(Dense2D(dim, hidden_dim; bias=false) for i = 1:3), Dense2D(hidden_dim, dim))
end

function (self::SelfAttention)(x)
    n, _, B = size(x)

    q, k, v = (reshape(self.qkv[i](x), n, :, self.heads, B) .* self.scale for i = 1:3)

    k = softmax(k, dims=1)
    
    context = ein"n d h b, n e h b -> e d h b"(k, v)

    out = ein"e d h b, n d h b -> n e h b"(context, q)
    out = reshape(out, n, :, B)

    return self.out(out)
end

struct TransformerDDPM
    layers::Chain
    pos_emb::AbstractArray
    num_attention_layers::Int64
    num_mlp_layers::Int64
end

@functor TransformerDDPM

function TransformerDDPM(dim=512, heads=8, attention_layers=6, mlp_layers=2, mlp_dims=2048, channels=32, emb_dim=dim)
    pos_emb = copy(reshape(SinusoidalPosEmb(emb_dim)(1:channels)', channels, emb_dim, 1))
    layers = []
    push!(layers, Dense2D(dim, emb_dim))

    for i = 1:attention_layers
        push!(layers, NDBatchNorm(emb_dim))
        push!(layers, SelfAttention(emb_dim, heads))
        push!(layers, NDBatchNorm(emb_dim))
        push!(layers, Dense2D(emb_dim, mlp_dims, gelu))
        push!(layers, Dense2D(mlp_dims, emb_dim, gelu))
    end

    push!(layers, NDBatchNorm(emb_dim))
    push!(layers, Dense2D(emb_dim, mlp_dims))

    for i = 1:mlp_layers
        push!(layers, DenseFiLM(128, mlp_dims))
        push!(layers, DenseResBlock(mlp_dims, mlp_dims, channels))
    end

    push!(layers, NDBatchNorm(mlp_dims))
    push!(layers, Dense2D(mlp_dims, dim))
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

#todo: remove dbg
#att = SelfAttention(64, 4, 32) |> gpu
#whcb
#att(ones(32, 64, 64) |> gpu)

#att = SelfAttention(512, 8, 128) |> gpu
#att(ones(32, 512, 64) |> gpu)

#println("model_params: ", count_params(att))

#todo: remove dbg

#ddpm = TransformerDDPM() |> gpu
#x = ones(32, 512, 21) |> gpu
#t = ones(21) |> gpu
#dbg_dump_var(x, "x")
#y = ddpm(x, t)
#dbg_dump_var(y, "y")