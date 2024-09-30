using SymbolicRegression
using Symbolics
using Distributions
using Optimization, OptimizationBBO
using Plots
using Random;
Random.seed!(12345);

y(x) = exp(-x) * sin(2π * x) + cos(π / 2 * x)
y(0.0)

n_obs = 10
design_region = Uniform(0.0, 10.0)
X = rand(design_region, n_obs)
Y = y.(X)
plot(0.0:0.1:10.0, y.(0.0:0.1:10.0), label="true model", lw=5, ls=:dash);
scatter!(X, Y, ms=5, label="data");
plot!(xlabel="x", ylabel="y", ylims=(-1.2, 1.8));
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=8, grid=false, dpi=600)

options = SymbolicRegression.Options(
    unary_operators=(exp, sin, cos),
    binary_operators=(+, *, /, -),
    seed=123,
    deterministic=true,
    save_to_file=false
)
hall_of_fame = equation_search(X', Y; options, niterations=100, runtests=false, parallelism=:serial)
n_best_max = 10
#incase < 10 model structures were returned
n_best = min(length(hall_of_fame.members), n_best_max)
best_models = sort(hall_of_fame.members, by=member -> member.loss)[1:n_best]

@syms x
eqn = node_to_symbolic(best_models[1].tree, options, variable_names=["x"])

f = build_function(eqn, x, expression=Val{false})
f.(X)

plot(0.0:0.1:10.0, y.(0.0:0.1:10.0), lw=5, label="true model", ls=:dash);
model_structures = []
for i = 1:n_best
    eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
    fi = build_function(eqn, x, expression=Val{false})
    x_plot = Float64[]
    y_plot = Float64[]
    for x_try in 0.0:0.1:10.0
        try
            y_try = fi(x_try)
            append!(x_plot, x_try)
            append!(y_plot, y_try)
        catch
        end
    end
    plot!(x_plot, y_plot, label="model $i")
    push!(model_structures, fi)
end
scatter!(X, Y, ms=5, label="data", ls=:dash);
plot!(xlabel="x", ylabel="y", ylims=(-1.2, 1.6));
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=8, grid=false, dpi=600, legend=:topright)

function S_criterion(x, model_structures)
    n_structures = length(model_structures)
    n_obs = length(x)
    if length(model_structures) == 1
        # sometimes only a single model structure comes out of the equation search
        return 0.0
    end
    y = zeros(n_obs, n_structures)
    for i in 1:n_structures
        y[:, i] .= model_structures[i].(x)
    end
    squared_differences = Float64[]
    for i in 1:n_structures
        for j in i+1:n_structures
            push!(squared_differences, maximum([k for k in (y[:, i] .- y[:, j]) .^ 2]))
        end
    end
    -mean(squared_differences) # minus sign to minimize instead of maximize
end
function S_objective(x_new, (x_old, model_structures))
    S_criterion([x_old; x_new], model_structures)
end
n_batch = 3
X_new_ini = rand(design_region, n_batch)
S_objective(X_new_ini, (X, model_structures))

lb = fill(minimum(design_region), n_batch)
ub = fill(maximum(design_region), n_batch)
prob = OptimizationProblem(S_objective, X_new_ini, (X, model_structures), lb=lb, ub=ub)
X_new = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=10.0)

Y_new = y.(X_new)
plot(0.0:0.1:10.0, y.(0.0:0.1:10.0), lw=5, label="true model", ls=:dash);
for i = 1:n_best
    x_plot = Float64[]
    y_plot = Float64[]
    for x_try in 0.0:0.01:10.0
        try
            y_try = model_structures[i](x_try)
            append!(x_plot, x_try)
            append!(y_plot, y_try)
        catch
        end
    end
    plot!(x_plot, y_plot, label="model $i")
end
scatter!(X, Y, ms=5, label="data old");
scatter!(X_new, Y_new, ms=5, label="data new");
plot!(xlabel="x", ylabel="y", ylim=(-1.2, 1.8));
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=8, grid=false, dpi=600)

X = [X; X_new]
Y = [Y; Y_new]
hall_of_fame = equation_search(X', Y, options=options, niterations=100, runtests=false, parallelism=:serial)
n_best = min(length(hall_of_fame.members), n_best_max)
best_models = sort(hall_of_fame.members, by=member -> member.loss)[1:n_best]
plot(0.0:0.01:10.0, y.(0.0:0.01:10.0), lw=5, label="true model", ls=:dash);
model_structures = []
for i = 1:n_best
    eqn = node_to_symbolic(best_models[i].tree, options, variable_names=["x"])
    println(eqn)
    fi = build_function(eqn, x, expression=Val{false})
    x_plot = Float64[]
    y_plot = Float64[]
    for x_try in 0.0:0.01:10.0
        try
            y_try = fi(x_try)
            append!(x_plot, x_try)
            append!(y_plot, y_try)
        catch
        end
    end
    plot!(x_plot, y_plot, label="model $i")
    push!(model_structures, fi)
end
scatter!(X, Y, ms=5, label="data");
plot!(xlabel="x", ylabel="y", ylims=(-1.2, 1.8));
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=8, grid=false, dpi=600)
