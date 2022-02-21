function dbg_dump_var(var, name)
    println("$name: ($(typeof(var))) [$(size(var))]")
end