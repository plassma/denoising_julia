using Flux

function dbg_dump_var(var, name)
    println("$name: ($(typeof(var))) [$(size(var))]")
end

function count_params(model)
    sum(length, params(model))
end

function showall(x, io=stdout, limit = false) 
    println(io, summary(x), ":")
    Base.print_matrix(IOContext(io, :limit => limit), x)
    println()
  end

function count_params_per_layer(model::Chain)
    for l in model
        println("$(typeof(l)): $(sum(length, params(l)))")
    end
end