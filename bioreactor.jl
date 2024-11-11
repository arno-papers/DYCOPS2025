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
using Statistics
using DataFrames

optimization_state =  [2.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
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
        C_s(t) = 3.0
        C_x(t) = 0.25
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
        K_s = 0.439
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
        chain = multi_layer_feed_forward(1, 1)
    end
    @components begin
#=         nn_in = RealInputArray(nin=1)
        nn_out = RealOutputArray(nout=1) =#
        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain)
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

sd_cs = 1
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

res = solve(op, Optimization.LBFGS(), maxiters=1000, callback=plot_cb)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())
plot(res_sol; label=["Cₛ(g/L) trained" "Cₓ(g/L) trained" "V(L) trained"], xlabel="t(h)", lw=3);
scatter!(data[!, "timestamp"], data[!, "C_s(t)"]; label=["Cₛ(g/L) true",], ms=3);
scatter!(data[!, "timestamp"], data[!, "C_x(t)"]; label=["Cₓ(g/L) true",], ms=3);
scatter!(data[!, "timestamp"], data[!, "V(t)"]; label=["V(L) true"], ms=3);


plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600,legend=false)

## get chain from the equations
extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
C_s = LuxCore.stateless_apply(extracted_chain, [20.0],convert(T,res.u))
C_s_range = 0.0:0.1:40.0 # do something more elegant than 0 .. 40 later, e.g. 100 steps between max and min of res_sol
μ_predicted = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range]
plot( 0.0:0.1:40.0, μ_predicted )
μ_max = 0.421
K_s = 0.439
plot!(C_s_range, μ_max .* C_s_range ./ (K_s .+ C_s_range))
## get plausible model structures for missing physics
options = SymbolicRegression.Options(
    unary_operators=(exp, sin, cos),
    binary_operators=(+, *, /, -),
    seed=123,
    deterministic=true,
    save_to_file=false
)
hall_of_fame = equation_search(collect(C_s_range)', μ_predicted; options, niterations=100, runtests=false, parallelism=:serial)

n_best_max = 10
n_best = min(length(hall_of_fame.members), n_best_max) #incase < 10 model structures were returned
best_models = sort(hall_of_fame.members, by=member -> member.loss)[1:n_best]

@syms x
eqn = node_to_symbolic(best_models[1].tree, options, variable_names=["x"])

f = build_function(eqn, x, expression=Val{false})
f.(collect(C_s_range)')

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
plot!()

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
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

## simulate with different controls

optimization_state =  [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
plot(sol ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
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
    plausible_prob = ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, saveat = 0:15)
    probs_plausible[i] = plausible_prob

    callback_controls = plausible_bioreactor.controls
    initial_control = plausible_bioreactor.Q_in
    syms_cache[i] = (callback_controls, initial_control)

    plausible_prob.ps[callback_controls] = optimization_state[2:end]
    plausible_prob.ps[initial_control] = optimization_state[1]

    plausible_sol = solve(plausible_prob, Rodas5P())
    plot!(plausible_sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)
# optimize the control pars

function S_criterion(optimization_state, (probs_plausible, syms_cache))
    n_structures = length(probs_plausible)
    if n_structures == 1
        # sometimes only a single model structure comes out of the equation search
        return error("Only a single model structure.")
    end
    sols = Array{Any}(undef, n_structures)
    for i in 1:n_structures
        plausible_prob = probs_plausible[i]
        callback_controls, initial_control  = syms_cache[i]
        plausible_prob.ps[callback_controls] = optimization_state[2:end]
        plausible_prob.ps[initial_control] = optimization_state[1]
        sol_plausible = solve(plausible_prob, Rodas5P())
        sols[i] = sol_plausible
    end
    squared_differences = Float64[]
    for i in 1:n_structures
        for j in i+1:n_structures
            push!(squared_differences, maximum((sols[i][1,:] .-sols[j][1,:]).^ 2)) # hardcoded first state, should be symbolic
        end
    end
    ret = -minimum(squared_differences) # minus sign to minimize instead of maximize
    # try mean instead of minimum...
    println(ret)
    return ret
end
S_criterion(zeros(15), (probs_plausible, syms_cache))

lb = zeros(15)
ub = 10*ones(15)
prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=2.0)

plot()
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control  = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible  = solve(plausible_prob , Rodas5P())
    plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600,legend=false)
