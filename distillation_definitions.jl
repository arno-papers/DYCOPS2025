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

# Simple Rayleigh batch distillation
# dL/dt = -V(t)                                    (liquid holdup decreasing)
# d(L*x)/dt = -V(t)*y(x)                           (component balance)
# => dx/dt = V(t)/L * (x - y(x))                   (after expanding)
# where y = phi(x) is the unknown VLE (vapor composition as function of liquid composition)
#
# True VLE: modified Raoult's law with Margules activity coefficients
# For a more interesting shape, we use a non-ideal mixture:
# y(x) = alpha(x)*x / (1 + (alpha(x)-1)*x)
# with constant relative volatility alpha for the simple case:
# y(x) = alpha*x / (1 + (alpha-1)*x)

@mtkmodel BatchDistillation begin
    @parameters begin
        controls[1:length(optimization_state)-1] = optimization_state[2:end], [tunable = false]
        V_rate = optimization_initial, [tunable = false] # boilup rate (mol/h)
    end
    @variables begin
        x(t) = 0.5     # liquid mole fraction of light component
        L(t) = 100.0    # liquid holdup (mol)
        y(t)            # vapor composition (missing physics)
    end
    @equations begin
        D(L) ~ -V_rate
        D(x) ~ V_rate / L * (x - y)
    end
    @discrete_events begin
        (t == 1.0) => [V_rate ~ controls[1]]
        (t == 2.0) => [V_rate ~ controls[2]]
        (t == 3.0) => [V_rate ~ controls[3]]
        (t == 4.0) => [V_rate ~ controls[4]]
        (t == 5.0) => [V_rate ~ controls[5]]
        (t == 6.0) => [V_rate ~ controls[6]]
        (t == 7.0) => [V_rate ~ controls[7]]
        (t == 8.0) => [V_rate ~ controls[8]]
        (t == 9.0) => [V_rate ~ controls[9]]
        (t == 10.0) => [V_rate ~ controls[10]]
        (t == 11.0) => [V_rate ~ controls[11]]
        (t == 12.0) => [V_rate ~ controls[12]]
        (t == 13.0) => [V_rate ~ controls[13]]
        (t == 14.0) => [V_rate ~ controls[14]]
        (t == 15.0) => [V_rate ~ optimization_initial] # HACK TO GET CONTROL BACK TO ITS ORIGINAL VALUE
    end
end
@mtkmodel TrueBatchDistillation begin
    @extend BatchDistillation()
    @parameters begin
        α = 3.0   # relative volatility (non-ideal mixture, moderate separation)
    end
    @equations begin
        y ~ α * x / (1 + (α - 1) * x)  # constant relative volatility VLE
    end
end
@mtkmodel UDEBatchDistillation begin
    @extend BatchDistillation()
    @structural_parameters begin
        chain = Lux.Chain(Lux.Dense(1, 5, tanh),
                          Lux.Dense(5, 5, tanh),
                          Lux.Dense(5, 1, sigmoid)) # output bounded [0,1] for mole fraction
    end
    @components begin
        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)
    end
    @equations begin
        nn.output.u[1] ~ y
        nn.input.u[1] ~ x
    end
end

α_true = 3.0
y_vle(x) = α_true * x / (1 + (α_true - 1) * x)

sd_x = 0.01  # measurement noise on composition (tighter for mole fractions)

function loss(x_params, (probs, get_varss, datas))
    loss = zero(eltype(x_params))
    for i in eachindex(probs)
        prob = probs[i]
        get_vars = get_varss[i]
        data = datas[i]
        new_p = SciMLStructures.replace(Tunable(), prob.p, x_params)
        new_prob = remake(prob, p=new_p, u0=eltype(x_params).(prob.u0))
        new_sol = solve(new_prob, Rodas5P())
        for (i, j) in enumerate(1:2:length(new_sol.t)) # HACK TO DEAL WITH DOUBLE SAVE
            loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "x(t)"][i]))
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
    @syms z
    for i = 1:n_best
        eqn = node_to_symbolic(best_models[i].tree, options, varMap=["z"])
        fi = build_function(eqn, z, expression=Val{false})
        push!(model_structures, fi)
    end
    return model_structures
end

function get_probs_and_caches(model_structures)
    probs_plausible = Array{Any}(undef, length(model_structures))
    syms_cache = Array{Any}(undef, length(model_structures))
    i = 1
    for i in 1:length(model_structures)
        @mtkmodel PlausibleDistillation begin
            @extend BatchDistillation()
            @equations begin
                y ~ model_structures[i](x)
            end
        end
        @mtkbuild plausible_dist = PlausibleDistillation()
        plausible_prob = ODEProblem(plausible_dist, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
        probs_plausible[i] = plausible_prob

        callback_controls = plausible_dist.controls
        initial_control = plausible_dist.V_rate

        syms_cache[i] = (callback_controls, initial_control, plausible_dist.x)
    end
    probs_plausible, syms_cache
end

function S_criterion(optimization_state, (probs_plausible, syms_cache))
    n_structures = length(probs_plausible)
    sols = Array{Any}(undef, n_structures)
    for i in 1:n_structures
        plausible_prob = probs_plausible[i]
        callback_controls, initial_control, x_var = syms_cache[i]
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
        callback_controls, initial_control, x_var = syms_cache[i]
        for j in i+1:n_structures
            push!(squared_differences, maximum((sols[i][x_var] .- sols[j][x_var]) .^ 2))
        end
    end
    ret = -mean(squared_differences)
    println(ret)
    return ret
end


lb = zeros(15)
ub = 5.0 * ones(15)  # boilup rate bounds (mol/h), must keep L > 0 over 15h (L0=100, max drain = 5*15=75)
