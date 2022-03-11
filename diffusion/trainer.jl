using Flux, Flux.Optimise, Zygote, CUDA
using Flux.Data: DataLoader
using Dates, Plots, Printf, ProgressBars, Statistics
using BSON: @save


struct Trainer
    diffusion_model#::GaussianDiffusionModel
    dataloader#::DataLoader

    train_lr#::Float64
    epochs#::Int64
    test_dataloader
end

function plot_losses(train_losses, test_losses, path)
    p = plot(train_losses, title="Diffusion Model training", label="train loss", xlabel="Epoch", ylabel="Mean Squared Error")

    if length(test_losses) == length(test_losses)
        plot!(test_losses, label="test loss")
    end
    display(p)
    savefig(p, path)
end

function loss_increased_for_n_epochs(test_losses, n = 2)
    if length(test_losses) > n
        stop=true
        for i = 1:n
            stop &= test_losses[end - n] < test_losses[end - i + 1]
        end
        return stop
    end
    false
end

function train!(trainer::Trainer; save_model=true, handle_samples=nothing, early_stopping_criterion=loss_increased_for_n_epochs)

    save_path = "results/$(DateTime(now()))/"
    mkpath(save_path)

    opt = Optimiser(ADAM(trainer.train_lr), ClipValue(1e-3), WeightDecay(1f-4))

    train_losses, test_losses = Vector{Float64}(), Vector{Float64}()

    move_x = (typeof(trainer.diffusion_model.device([1])) <: CUDA.CuArray) != (typeof(trainer.dataloader.data) <: CUDA.CuArray)
    for epoch = 1:trainer.epochs
        losses = Vector{Float64}()

        trainmode!(trainer.diffusion_model)
        
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
            set_description(iter, string(@sprintf("train loss: %.5f", loss)))
        end

        push!(train_losses, mean(losses))
        println("Epoch $epoch/$(trainer.epochs), train loss: $(train_losses[end])")
        testmode!(trainer.diffusion_model)
        
        if !isnothing(trainer.test_dataloader)
            iter = ProgressBar(trainer.test_dataloader)
            losses = Vector{Float64}()
            for x in iter
                loss = move_x ? trainer.diffusion_model(x |> trainer.diffusion_model.device) : trainer.diffusion_model(x)
                push!(losses, loss)
                set_description(iter, string(@sprintf("test loss: %.5f", loss)))
            end
            push!(test_losses, mean(losses))
            println("Epoch $epoch/$(trainer.epochs), test loss: $(test_losses[end])")
        end

        if !isnothing(handle_samples)
            epoch_samples = sample(trainer.diffusion_model)
            handle_samples(epoch_samples |> cpu, save_path)
        end

        if save_model
            model = trainer.diffusion_model |> cpu
            path = save_path * "diffusion_model_" * string(@sprintf("%04d", epoch)) * ".bson"
            @save path model
        end

        if early_stopping_criterion(test_losses)
            println("Early stopping criterion met!")
            break
        end
    end
    plot_losses(train_losses, test_losses, save_path)
end
