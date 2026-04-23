using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitNeuralNets
using OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface
using Plots
using Optimization, OptimizationOptimisers, OptimizationBBO, OptimizationNLopt
using SciMLStructures
using SciMLStructures: Tunable
using SciMLSensitivity
using Statistics
using SymbolicRegression
using LuxCore
using LuxCore: stateless_apply
using Lux
using Statistics
using DataFrames

@mtkmodel CSTR begin
    @constants begin
        C_s_in = 20.0
        y_x_s = 0.5
        m = 0.0
    end
    @parameters begin
        controls[1:length(optimization_state)-1] = optimization_state[2:end], [tunable = false]
        Q_in = optimization_initial, [tunable = false]
    end
    @variables begin
        C_s(t) = 1.0
        C_x(t) = 0.5
        μ(t)
        σ(t)
    end
    @equations begin
        σ ~ μ / y_x_s + m
        D(C_s) ~ -σ * C_x + Q_in * (C_s_in - C_s)
        D(C_x) ~ μ * C_x - Q_in * C_x
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
@mtkmodel TrueCSTR begin
    @extend CSTR()
    @parameters begin
        μ_max = 1.0
        K_s = 1.0
        K_i = 10.0
    end
    @equations begin
        μ ~ μ_max * C_s / (K_s + C_s + C_s^2 / K_i) # substrate inhibition (Andrews/Haldane)
    end
end
@mtkmodel UDECSTR begin
    @extend CSTR()
    @structural_parameters begin
        chain = Lux.Chain(Lux.Dense(1, 5, tanh),
                          Lux.Dense(5, 5, tanh),
                          Lux.Dense(5, 1, x->1*sigmoid(x)))
    end
    @components begin
        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)
    end
    @equations begin
        nn.output.u[1] ~ μ
        nn.input.u[1] ~ C_s
    end
end

μ_max = 1.0
K_s = 1.0
K_i = 10.0

sd_cs = 0.1

function loss(x, (probs, get_varss, datas))
    loss = zero(eltype(x))
    for i in eachindex(probs)
        prob = probs[i]
        get_vars = get_varss[i]
        data = datas[i]
        new_p = SciMLStructures.replace(Tunable(), prob.p, x)
        new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
        new_sol = solve(new_prob, Rodas5P())
        for (i, j) in enumerate(1:2:length(new_sol.t)) # HACK TO DEAL WITH DOUBLE SAVE
            loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "C_s(t)"][i]))
        end
        if !(SciMLBase.successful_retcode(new_sol))
            println("failed")
            return Inf
        end
    end
    println(loss)
    loss
end

options = SymbolicRegression.Options(
    unary_operators=(exp, sin, cos),
    binary_operators=(+, *, /, -),
    seed=123,
    deterministic=true,
    save_to_file=false
)

n_best = 10

function get_model_structures(hall_of_fame, options, n_best)
    best_models = []
    best_models_scores = []
    i = 1
    round(hall_of_fame.members[i].loss,sigdigits=5)
    while length(best_models) <= n_best
        member = hall_of_fame.members[i]
        rounded_score = round(member.loss, sigdigits=5)
        if !(rounded_score in best_models_scores)
            push!(best_models,member)
            push!(best_models_scores, rounded_score)
        end
        i += 1
    end
    model_structures = []
    @syms x
    for i = 1:n_best
        eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
        fi = build_function(eqn, x, expression=Val{false})
        push!(model_structures, fi)
    end
    return model_structures
end

function get_probs_and_caches(model_structures)
    probs_plausible = Array{Any}(undef, length(model_structures))
    syms_cache = Array{Any}(undef, length(model_structures))
    i = 1
    for i in 1:length(model_structures)
        @mtkmodel PlausibleCSTR begin
            @extend CSTR()
            @equations begin
                μ ~ model_structures[i](C_s)
            end
        end
        @mtkbuild plausible_cstr = PlausibleCSTR()
        plausible_prob = ODEProblem(plausible_cstr, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
        probs_plausible[i] = plausible_prob

        callback_controls = plausible_cstr.controls
        initial_control = plausible_cstr.Q_in

        syms_cache[i] = (callback_controls, initial_control, plausible_cstr.C_s)
    end
    probs_plausible, syms_cache
end

function S_criterion(optimization_state, (probs_plausible, syms_cache))
    n_structures = length(probs_plausible)
    sols = Array{Any}(undef, n_structures)
    for i in 1:n_structures
        plausible_prob = probs_plausible[i]
        callback_controls, initial_control, C_s = syms_cache[i]
        plausible_prob.ps[callback_controls] = optimization_state[2:end]
        plausible_prob.ps[initial_control] = optimization_state[1]
        sol_plausible = solve(plausible_prob, Rodas5P())
        if !(SciMLBase.successful_retcode(sol_plausible))
            return 0.0
        end
        sols[i] = sol_plausible
    end
    squared_differences = Float64[]
    for i in 1:n_structures
        callback_controls, initial_control, C_s = syms_cache[i]
        for j in i+1:n_structures
            push!(squared_differences, maximum((sols[i][C_s] .- sols[j][C_s]) .^ 2))
        end
    end
    ret = -mean(squared_differences)
    println(ret)
    return ret
end


lb = zeros(15)
ub = 0.5 * ones(15)
