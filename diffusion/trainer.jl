using Flux
using Flux.Data: DataLoader
using Zygote
using Statistics
using ProgressBars
using CUDA
using Printf
using BSON: @save
using Dates


struct Trainer
    diffusion_model#::GaussianDiffusionModel
    dataloader#::DataLoader

    train_lr#::Float64
    epochs#::Int64
end

function train(trainer::Trainer; save_model=true, plot=nothing)

    save_path = "results/$(DateTime(now()))/"
    mkpath(save_path)

    opt = ADAM(trainer.train_lr)

    move_x = (typeof(trainer.diffusion_model.device([1])) <: CUDA.CuArray) != (typeof(trainer.dataloader.data) <: CUDA.CuArray)

    for epoch = 1:trainer.epochs
        losses = Vector{Float64}()

        iter = ProgressBar(trainer.dataloader)
        for x in iter
            params = Flux.params(trainer.diffusion_model.denoise_fn)
            loss, back = Zygote.pullback(params) do
                loss = move_x ? trainer.diffusion_model(x |> trainer.diffusion_model.device) : trainer.diffusion_model(x)# todo: moving the data anywhere in the controlflow seems to create a copy slowing down the training by 50%. This is not elegant, but works
            end
            grads = back(1.0f0)
            Flux.update!(opt, params, grads)
            loss = loss |> cpu
            push!(losses, loss)
            set_description(iter, string(@sprintf("Loss: %.5f", loss)))
        end

        if !isnothing(plot)
            epoch_samples = sample(trainer.diffusion_model)
            plot(epoch_samples |> cpu)
        end
        
        if save_model
            model = trainer.diffusion_model |> cpu
            path = save_path * "diffusion_model_" * string(@sprintf("%04d", epoch)) * ".bson"
            @save path model
        end
        println("Epoch $epoch/$(trainer.epochs),  loss: $(mean(losses))")
    end
end