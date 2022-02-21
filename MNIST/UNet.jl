module UNet

export Unet, Fake2DUnet, Fake1DUnet
export SinusoidalPosEmb #todo: dbg exports
using Flux
using Flux: @functor
using Distributions: Normal

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)

function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

struct SinusoidalPosEmb
    emb::AbstractArray{Float32}
end

@functor SinusoidalPosEmb

SinusoidalPosEmb(dim::Int64) = SinusoidalPosEmb(ℯ .^ ((0:(dim÷2-1)) * -(log(10000) / (dim ÷ 2 - 1))))

function (s::SinusoidalPosEmb)(x)
    emb = x' .* s.emb
    return cat(sin.(emb), cos.(emb), dims = 1)
end


function BatchNormWrap(out_ch)
    Chain(x -> expand_dims(x, 2),
        BatchNorm(out_ch),
        x -> squeeze(x))
end

struct ConvBlock
    time_mlp
    ds_conv
    chain
    res_conv
end

@functor ConvBlock

ConvBlock(dim::Int, dim_out::Int, norm::Bool = true, time_emb_dim::Int = 64, mult::Int = 2) = ConvBlock(
    Chain(x -> gelu.(x), Dense(time_emb_dim, dim)),
    Conv((7, 7), dim => dim, pad = (3, 3), groups = dim),
    Chain(norm ? BatchNormWrap(dim) : x -> x,
        Conv((3, 3), dim => dim_out * mult, pad=(1, 1)), x -> gelu.(x),
        Conv((3, 3), dim_out * mult => dim_out, pad=(1, 1))),
    dim == dim_out ? x -> x : Conv((1, 1), dim => dim_out)
    )  

function (u::ConvBlock)(x, t = nothing)
    h = u.ds_conv(x)
    if ! isnothing(t)
        t = u.time_mlp(t)
        resh = reshape(t, (1, 1, size(t)...))
        h = h .+ resh
    end
    h = u.chain(h)
    return h + u.res_conv(x)
end

Upsample(dim) = ConvTranspose((4, 4), dim => dim, stride=(2, 2), pad=(1, 1))
Downsample(dim) = Conv((4, 4), dim => dim, stride=(2, 2), pad=(1, 1))


struct Unet
    down_blocks
    conv_blocks
    up_blocks
    time_mlp
    final_conv
end

@functor Unet

function Unet(channels::Int = 1, out_dim::Int = channels, time_dim::Int = 64)
    down_blocks = Chain(Downsample(64))

    conv_blocks = Chain(ConvBlock(1, 64, false), ConvBlock(64, 64), ConvBlock(64, 128), ConvBlock(128, 128), ConvBlock(128, 128), ConvBlock(128, 128),
                        ConvBlock(256, 64), ConvBlock(64, 64))

    up_blocks = Chain(Upsample(64))

    time_mlp = Chain(SinusoidalPosEmb(time_dim),
        Dense(time_dim, time_dim * 4, gelu),
        Dense(time_dim * 4, time_dim))
    
    final_conv = Chain(ConvBlock(64, 64), Conv((1, 1), 64 => out_dim))

    Unet(down_blocks, conv_blocks, up_blocks, time_mlp, final_conv)
end

function (u::Unet)(x::AbstractArray, t::AbstractArray)

    t = u.time_mlp(t)
    
    x = u.conv_blocks[1](x, t)
    x = u.conv_blocks[2](x, t)
    h1 = x
    x = u.down_blocks[1](x)

    x = u.conv_blocks[3](x, t)
    x = u.conv_blocks[4](x, t)
    h2 = x

    x = u.conv_blocks[5](x, t)
    x = u.conv_blocks[6](x, t)

    x = u.conv_blocks[7](cat(x, h2, dims = 3), t)
    x = u.conv_blocks[8](x, t)
    x = u.up_blocks[1](x)

    return u.final_conv(x)
  end


struct Fake2DUnet
    unet::Unet
    dim
end

@functor Fake2DUnet

Fake2DUnet(dim) = Fake2DUnet(Unet(), dim)

function (f::Fake2DUnet)(x::AbstractArray, t::AbstractArray)
    shape = size(x)
    x = reshape(x, (f.dim, f.dim, size(x)[2:end]...))
    x = f.unet(x, t)
    return reshape(x, shape)
end


struct Fake1DUnet
    unet::Unet
    dim
end

@functor Fake1DUnet

Fake1DUnet(dim) = Fake1DUnet(Unet(), dim)

function (f::Fake1DUnet)(x::AbstractArray, t::AbstractArray)
    shape = size(x)
    x = reshape(x, (f.dim, f.dim, 1, size(x)[2:end]...))
    x = f.unet(x, t)
    return reshape(x, shape)
end


end #module