using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitNeuralNets
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEqTsit5, OrdinaryDiffEqNonlinearSolve, OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface
using Plots
using Optimization, OptimizationOptimisers, OptimizationBBO
using SciMLStructures
using SciMLStructures: Tunable
using SciMLSensitivity
using Statistics
using SymbolicRegression
using LuxCore
using Lux
using Statistics
using DataFrames
using Random; Random.seed!(984519674645)
using StableRNGs

rng = StableRNG(123)

function plot3!(plts, sol)
    plot!(plts[1], sol, idxs=:C_s, title="Cₛ(g/L)", xlabel="t(h)", lw=3)
    plot!(plts[2], sol, idxs=:C_x, title="Cₓ(g/L)", xlabel="t(h)", lw=3)
    plot!(plts[3], sol, idxs=:V, title="V(L)", xlabel="t(h)", lw=3)
end

optimization_state =  zeros(15)
optimization_initial = optimization_state[1]
@mtkmodel Bioreactor begin
    @constants begin
        C_s_in = 50.0
        y_x_s = 0.777
        m = 0.0
    end
    @parameters begin
        controls[1:length(optimization_state)-1] = optimization_state[2:end], [tunable = false] # zero vector
        Q_in = optimization_initial, [tunable = false] # zero value # make the initial parameter value the first value of the control array, can't get it to work
    end
    @variables begin
        C_s(t) = 1.0
        C_x(t) = 1.0
        V(t) = 7.0
        μ(t)
        σ(t)
    end
    @equations begin
        σ ~ μ / y_x_s + m
        D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
        D(C_x) ~ μ * C_x - Q_in / V * C_x
        D(V) ~ Q_in
    end
    @discrete_events begin
        (t == 1.0) => [Q_in ~ controls[1]]
        (t == 2.0) => [Q_in ~ controls[2]]
        (t == 3.0) => [Q_in ~ controls[3]]
        (t == 4.0) => [Q_in ~ controls[4]]
        (t == 5.0) => [Q_in ~ controls[5]]
        (t == 6.0) => [Q_in ~ controls[6]]
        (t == 7.0) => [Q_in ~ controls[7]]
        (t == 8.0) => [Q_in ~ controls[8]]
        (t == 9.0) => [Q_in ~ controls[9]]
        (t == 10.0) => [Q_in ~ controls[10]]
        (t == 11.0) => [Q_in ~ controls[11]]
        (t == 12.0) => [Q_in ~ controls[12]]
        (t == 13.0) => [Q_in ~ controls[13]]
        (t == 14.0) => [Q_in ~ controls[14]]
        (t == 15.0) => [Q_in ~ optimization_initial] # HACK TO GET Q_IN BACK TO ITS ORIGINAL VALUE
    end
end
@mtkmodel TrueBioreactor begin
    @extend Bioreactor()
    @parameters begin
        μ_max = 0.421
        K_s = 0.439*10
    end
    @equations begin
        μ ~ μ_max * C_s / (K_s + C_s) # this should be recovered from data
    end
end
@mtkbuild true_bioreactor = TrueBioreactor()
prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = solve(prob, Rodas5P())
plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

@mtkmodel UDEBioreactor begin
    @extend Bioreactor()
    @structural_parameters begin
        chain = Lux.Chain(Lux.Dense(1, 5, tanh),
                          Lux.Dense(5, 5, tanh),
                          Lux.Dense(5, 1, x->1*sigmoid(x)))
    end
    @components begin
#=         nn_in = RealInputArray(nin=1)
        nn_out = RealOutputArray(nout=1) =#
        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)
    end
    @equations begin
        nn.output.u[1] ~ μ
        nn.input.u[1] ~ C_s
#=         μ ~ nn_out.u[1]
        nn_in.u[1] ~ C_s
        connect(nn_in, nn.input)
        connect(nn_out, nn.output) =#
    end
end

@mtkbuild  ude_bioreactor = UDEBioreactor()

ude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = solve(ude_prob, Rodas5P())
plot(ude_sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

# we use all tunable parameters because it's easier on the remake
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))

get_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])
# TODO: Switch to data with noise instead of comparing against the reference sol
data = DataFrame(sol)
data = data[1:2:end, :]

sd_cs = 0.1
data[!, "C_s(t)"] += sd_cs * randn(size(data, 1))

plot(sol)
scatter!(data[!, "timestamp"], data[!, "C_s(t)"]; label="Cₛ(g/L) true", ms=3);
scatter!(data[!, "timestamp"], data[!, "C_x(t)"]; label="Cₓ(g/L) true", ms=3);
scatter!(data[!, "timestamp"], data[!, "V(t)"]; label="V(L) true", ms=3)

# get_refs = getu(true_bioreactor, [true_bioreactor.V, true_bioreactor.C_s])

function loss(x, (prob, sol_ref, get_vars, data))
    new_p = SciMLStructures.replace(Tunable(), prob.p, x)
    new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Rodas5P())
    loss = zero(eltype(x))
    for (i, j) in enumerate(1:2:length(new_sol.t))
        # @info "i: $i j: $j"
        loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "C_s(t)"][i]))
    end

    if SciMLBase.successful_retcode(new_sol)
        loss
    else
        Inf
    end

    loss
end

of = OptimizationFunction{true}(loss, AutoZygote())
ps = (ude_prob, sol, get_vars, data);

op = OptimizationProblem(of, x0, ps)
of(x0, ps)[1]

plot_cb = (opt_state, loss,) -> begin
    @info "step $(opt_state.iter), loss: $loss"
    # @info opt_state.u

    new_p = SciMLStructures.replace(Tunable(), ude_prob.p, opt_state.u)
    new_prob = remake(ude_prob, p = new_p)
    sol = solve(new_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol)
        display(plot(sol))
    end
    false
end

res = solve(op, Optimization.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())
plot(res_sol; label=["Cₛ(g/L) trained" "Cₓ(g/L) trained" "V(L) trained"], xlabel="t(h)", lw=3);
scatter!(data[!, "timestamp"], data[!, "C_s(t)"]; label=["Cₛ(g/L) true",], ms=3);
scatter!(data[!, "timestamp"], data[!, "C_x(t)"]; label=["Cₓ(g/L) true",], ms=3);
scatter!(data[!, "timestamp"], data[!, "V(t)"]; label=["V(L) true"], ms=3);



## get chain from the equations
extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
C_s = LuxCore.stateless_apply(extracted_chain, [20.0],convert(T,res.u))
C_s_range = range(minimum(data[!, "C_s(t)"]),maximum(data[!, "C_s(t)"]),100)
C_s_range_plot = 0.0:0.01:50.0
C_s_train =
μ_predicted = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range]
μ_predicted_plot = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]

μ_max = 0.421
K_s = 0.439*10
plt = plot(C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot))
plot!(C_s_range_plot, μ_predicted_plot)
predicted_data = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
scatter!(data[!, "C_s(t)"],  predicted_data)
## get plausible model structures for missing physics
options = SymbolicRegression.Options(
    unary_operators=(exp, sin, cos),
    binary_operators=(+, *, /, -),
    seed=123,
    deterministic=true,
    save_to_file=false
)
hall_of_fame = equation_search(collect(data[!, "C_s(t)"])', predicted_data; options, niterations=100, runtests=false, parallelism=:serial)

n_best_max = 10
n_best = min(length(hall_of_fame.members), n_best_max) #incase < 10 model structures were returned
best_models = []
best_models_scores = []
i = 1
round(hall_of_fame.members[i].loss,sigdigits=5)
while length(best_models) <= n_best
    println(i)
    member = hall_of_fame.members[i]
    rounded_score = round(member.loss, sigdigits=5)
    if !(rounded_score in best_models_scores)
        push!(best_models,member)
        push!(best_models_scores, rounded_score)
    end
    i += 1
end
best_models

@syms x
eqn = node_to_symbolic(best_models[1].tree, options, variable_names=["x"])

f = build_function(eqn, x, expression=Val{false})
f.(collect(C_s_range)')

function get_model_structures(best_models, options, n_best, plt)
    model_structures = []
    @syms x

    for i = 1:n_best
        eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
        fi = build_function(eqn, x, expression=Val{false})
        x_plot = Float64[]
        y_plot = Float64[]
        for x_try in C_s_range_plot
            try
                y_try = fi(x_try)
                append!(x_plot, x_try)
                append!(y_plot, y_try)
            catch
            end
        end
        plot!(plt, x_plot, y_plot, label="model $i")
        push!(model_structures, fi)
    end
    plot!(plt, legend=false,ylims=(0,1))

    return model_structures
end

model_structures = get_model_structures(best_models, options, n_best, plt)
plt

## get complete plausible model structures
plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
for i in 1:length(model_structures)
    @mtkmodel PlausibleBioreactor begin
        @extend Bioreactor()
        @equations begin
            μ ~ model_structures[i](C_s)
        end
    end
    @mtkbuild plausible_bioreactor = PlausibleBioreactor()
    plausible_prob = ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, saveat = 0:15)
    plausible_sol = solve(plausible_prob, Rodas5P())
    plot!(plausible_sol ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600, legend=false)

## simulate with different controls

optimization_state = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
probs_plausible = Array{Any}(undef, length(model_structures))
syms_cache = Array{Any}(undef, length(model_structures))
for i in 1:length(model_structures)
    @mtkmodel PlausibleBioreactor begin
        @extend Bioreactor()
        @equations begin
            μ ~ model_structures[i](C_s)
        end
    end
    @mtkbuild plausible_bioreactor = PlausibleBioreactor()
    plausible_prob = ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
    probs_plausible[i] = plausible_prob

    callback_controls = plausible_bioreactor.controls
    initial_control = plausible_bioreactor.Q_in

    syms_cache[i] = (callback_controls, initial_control, plausible_bioreactor.C_s)

    plausible_prob.ps[callback_controls] = optimization_state[2:end]
    plausible_prob.ps[initial_control] = optimization_state[1]

    plausible_sol = solve(plausible_prob, Rodas5P())
    plot!(plausible_sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600, legend=false)
# optimize the control pars

function S_criterion(optimization_state, (probs_plausible, syms_cache))
    try
    n_structures = length(probs_plausible)
    n_structures = length(probs_plausible)
    if n_structures == 1
        # sometimes only a single model structure comes out of the equation search
        return error("Only a single model structure.")
    end
        n_structures = length(probs_plausible)
    if n_structures == 1
        # sometimes only a single model structure comes out of the equation search
        return error("Only a single model structure.")
    end
        sols = Array{Any}(undef, n_structures)
        for i in 1:n_structures
            plausible_prob = probs_plausible[i]
            callback_controls, initial_control, C_s = syms_cache[i]
            plausible_prob.ps[callback_controls] = optimization_state[2:end]
            plausible_prob.ps[initial_control] = optimization_state[1]
            sol_plausible = solve(plausible_prob, Rodas5P())
            sols[i] = sol_plausible
        end
        squared_differences = Float64[]
        for i in 1:n_structures
            callback_controls, initial_control, C_s = syms_cache[i]
            for j in i+1:n_structures
                push!(squared_differences, maximum((sols[i][C_s] .- sols[j][C_s]) .^ 2)) # hardcoded first state, should be symbolic
            end
        end
        ret = -mean(squared_differences) # minus sign to minimize instead of maximize
        # try mean instead of minimum...
        println(ret)
        return ret
    catch e
        return 0.0
    end
end
S_criterion(zeros(15), (probs_plausible, syms_cache))

lb = zeros(15)
ub = 10 * ones(15)
prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)

S_criterion(control_pars_opt, (probs_plausible, syms_cache))
optimization_state = control_pars_opt.u
optimization_initial = optimization_state[1]

function plot_model_structures(model_structures, probs_plausible)
    plts = plot(), plot(), plot()
    for i in 1:length(model_structures)
        plausible_prob = probs_plausible[i]
        callback_controls, initial_control, C_s = syms_cache[i]
        plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
        plausible_prob.ps[initial_control] = control_pars_opt[1]
        sol_plausible = solve(plausible_prob, Rodas5P())
        # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
        plot3!(plts, sol_plausible)
    end
    plot(plts..., tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600, legend=false)
end

plot_model_structures(model_structures, probs_plausible)

@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = solve(prob2, Rodas5P())
plot(sol2; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

@mtkbuild ude_bioreactor2 = UDEBioreactor()

ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
ude_sol2 = solve(ude_prob2, Rodas5P())
plot(ude_sol2; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(sol2; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600,legend=false)

# we use all tunable parameters because it's easier on the remake
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor2),), tunable_parameters(ude_bioreactor2)))

get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])

data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))

function loss2(x, (prob1, prob2, get_vars1, get_vars2, data1, data2))
    new_p1 = SciMLStructures.replace(Tunable(), prob1.p, x)
    new_prob1 = remake(prob1, p=new_p1, u0=eltype(x).(prob1.u0))

    new_p2 = SciMLStructures.replace(Tunable(), prob2.p, x)
    new_prob2 = remake(prob1, p=new_p2, u0=eltype(x).(prob2.u0))

    new_sol1 = solve(new_prob1, Rodas5P())
    new_sol2 = solve(new_prob2, Rodas5P())


    loss = zero(eltype(x))
    for (i, j) in enumerate(1:2:length(new_sol1.t))
        # @info "i: $i j: $j"
        loss += sum(abs2.(get_vars1(new_sol1, j) .- data1[!, "C_s(t)"][i]))
        loss += sum(abs2.(get_vars2(new_sol2, j) .- data2[!, "C_s(t)"][i]))
    end

    if SciMLBase.successful_retcode(new_sol1) && SciMLBase.successful_retcode(new_sol2)
        loss
    else
        Inf
    end

    loss
end

of = OptimizationFunction{true}(loss2, AutoZygote())
ps = (ude_prob, ude_prob2, get_vars, get_vars2, data, data2);

op = OptimizationProblem(of, x0, ps)

res = solve(op, Optimization.LBFGS(), maxiters=1000)
new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
res_sol = solve(res_prob, Rodas5P())
plot(res_sol; label=["Cₛ(g/L) trained" "Cₓ(g/L) trained" "V(L) trained"], xlabel="t(h)", lw=3);
scatter!(data2[!, "timestamp"], data2[!, "C_s(t)"]; label=["Cₛ(g/L) true",], ms=3);
scatter!(data2[!, "timestamp"], data2[!, "C_x(t)"]; label=["Cₓ(g/L) true",], ms=3);
scatter!(data2[!, "timestamp"], data2[!, "V(t)"]; label=["V(L) true"], ms=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600, legend=false)

## get chain from the equations
extracted_chain = arguments(equations(ude_bioreactor2.nn)[1].rhs)[1]
T = defaults(ude_bioreactor2)[ude_bioreactor2.nn.T]
μ_predicted_plot2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]

plt = plot(C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot))
plot!(C_s_range_plot, μ_predicted_plot)
plot!(C_s_range_plot, μ_predicted_plot2)
predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
scatter!(data[!, "C_s(t)"],  predicted_data1)
scatter!(data2[!, "C_s(t)"],  predicted_data2)
total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'))
total_predicted_data =  vcat(predicted_data, predicted_data2)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=100, runtests=false, parallelism=:serial)
n_best_max = 10
n_best = min(length(hall_of_fame.members), n_best_max) #incase < 10 model structures were returned
best_models = []
best_models_scores = []
i = 1
round(hall_of_fame.members[i].loss,sigdigits=5)
while length(best_models) <= n_best
    println(i)
    member = hall_of_fame.members[i]
    rounded_score = round(member.loss, sigdigits=5)
    if !(rounded_score in best_models_scores)
        push!(best_models,member)
        push!(best_models_scores, rounded_score)
    end
    i += 1
end
best_models

# model_structures = []
# for i = 1:n_best
#     eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
#     fi = build_function(eqn, x, expression=Val{false})
#     x_plot = Float64[]
#     y_plot = Float64[]
#     for x_try in C_s_range_plot
#         try
#             y_try = fi(x_try)
#             append!(x_plot, x_try)
#             append!(y_plot, y_try)
#         catch
#         end
#     end
#     plot!(x_plot, y_plot, label="model $i")
#     push!(model_structures, fi)
# end
# plot!(legend=false)

model_structures = get_model_structures(best_models, options, n_best, plt)
plt

model_structures = []
probs_plausible = []
syms_cache = []
for i in 1:n_best
    eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
    fi = build_function(eqn, x, expression=Val{false})
    push!(model_structures, fi)
    @mtkmodel PlausibleBioreactor begin
        @extend Bioreactor()
        @equations begin
            μ ~ model_structures[i](C_s)
        end
    end
    @mtkbuild plausible_bioreactor = PlausibleBioreactor()
    plausible_prob = ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
    push!(probs_plausible, plausible_prob)
    callback_controls = plausible_bioreactor.controls
    initial_control = plausible_bioreactor.Q_in
    push!(syms_cache, (callback_controls, initial_control, plausible_bioreactor.C_s))
end

S_criterion(zeros(15), (probs_plausible, syms_cache))

prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)

S_criterion(control_pars_opt, (probs_plausible, syms_cache))
optimization_state = control_pars_opt.u
optimization_initial = optimization_state[1]
# plts = plot(), plot(), plot()
# for i in 1:length(model_structures)
#     plausible_prob = probs_plausible[i]
#     callback_controls, initial_control, C_s = syms_cache[i]
#     plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
#     plausible_prob.ps[initial_control] = control_pars_opt[1]
#     sol_plausible = solve(plausible_prob, Rodas5P())
#     # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
#     plot3!(plts, sol_plausible)
# end
# plot(plts..., tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600, legend=false)

plot_model_structures(model_structures, probs_plausible)
