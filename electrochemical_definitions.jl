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

# Simplified electrochemical cell
# The cell has a single reactant species with concentration C
# and an overpotential eta that determines reaction rate
#
# dC/dt = -j(eta)/(n*F*Vol) + k_supply     (reactant consumed by reaction, replenished at constant rate)
# d(eta)/dt = (U(t) - E_eq - eta) / (R_ohm * C_dl) - j(eta) / C_dl
#
# We simplify to focus on the Tafel regime (high overpotential),
# where the kinetics become a single exponential:
# j(eta) = j0 * exp(alpha * eta)    (anodic Tafel)
#
# This is recoverable by SR since exp is in the operator set.
# Full Butler-Volmer j = j0*(exp(alpha_a*eta) - exp(-alpha_c*eta)) is a stretch goal.

@mtkmodel ElectrochemicalCell begin
    @constants begin
        n_F_Vol = 1.0    # n*F/Vol lumped constant (1/V·s for normalized units)
        k_supply = 0.02  # constant reactant supply rate (mol/L/s normalized)
        R_ohm = 1.0      # ohmic resistance (Ohm, normalized)
        C_dl = 1.0       # double-layer capacitance (F, normalized)
        E_eq = 0.0       # equilibrium potential (V)
    end
    @parameters begin
        controls[1:length(optimization_state)-1] = optimization_state[2:end], [tunable = false]
        U_applied = optimization_initial, [tunable = false] # applied voltage
    end
    @variables begin
        C(t) = 1.0       # reactant concentration (normalized)
        eta(t) = 0.0     # overpotential (V)
        j(t)             # current density (missing physics)
    end
    @equations begin
        D(C) ~ -j * n_F_Vol + k_supply
        D(eta) ~ (U_applied - E_eq - eta) / (R_ohm * C_dl) - j / C_dl
    end
    @discrete_events begin
        (t == 1.0) => [U_applied ~ controls[1]]
        (t == 2.0) => [U_applied ~ controls[2]]
        (t == 3.0) => [U_applied ~ controls[3]]
        (t == 4.0) => [U_applied ~ controls[4]]
        (t == 5.0) => [U_applied ~ controls[5]]
        (t == 6.0) => [U_applied ~ controls[6]]
        (t == 7.0) => [U_applied ~ controls[7]]
        (t == 8.0) => [U_applied ~ controls[8]]
        (t == 9.0) => [U_applied ~ controls[9]]
        (t == 10.0) => [U_applied ~ controls[10]]
        (t == 11.0) => [U_applied ~ controls[11]]
        (t == 12.0) => [U_applied ~ controls[12]]
        (t == 13.0) => [U_applied ~ controls[13]]
        (t == 14.0) => [U_applied ~ controls[14]]
        (t == 15.0) => [U_applied ~ optimization_initial] # HACK TO GET CONTROL BACK TO ITS ORIGINAL VALUE
    end
end
@mtkmodel TrueElectrochemicalCell begin
    @extend ElectrochemicalCell()
    @parameters begin
        j0 = 0.01      # exchange current density
        α = 0.5         # charge transfer coefficient (dimensionless, scaled)
    end
    @equations begin
        j ~ j0 * exp(α * eta)  # Tafel kinetics (anodic branch)
    end
end
@mtkmodel UDEElectrochemicalCell begin
    @extend ElectrochemicalCell()
    @structural_parameters begin
        # NN output: use softplus to ensure j > 0 (current density is positive in anodic regime)
        # Scale factor accounts for range of j values
        chain = Lux.Chain(Lux.Dense(1, 5, tanh),
                          Lux.Dense(5, 5, tanh),
                          Lux.Dense(5, 1, softplus))
    end
    @components begin
        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)
    end
    @equations begin
        nn.output.u[1] ~ j
        nn.input.u[1] ~ eta
    end
end

j0_true = 0.01
α_true = 0.5
j_true(eta) = j0_true * exp(α_true * eta)

sd_C = 0.02  # measurement noise on concentration

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
            loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "C(t)"][i]))
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
        @mtkmodel PlausibleElectrochemical begin
            @extend ElectrochemicalCell()
            @equations begin
                j ~ model_structures[i](eta)
            end
        end
        @mtkbuild plausible_ec = PlausibleElectrochemical()
        plausible_prob = ODEProblem(plausible_ec, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
        probs_plausible[i] = plausible_prob

        callback_controls = plausible_ec.controls
        initial_control = plausible_ec.U_applied

        syms_cache[i] = (callback_controls, initial_control, plausible_ec.C)
    end
    probs_plausible, syms_cache
end

function S_criterion(optimization_state, (probs_plausible, syms_cache))
    n_structures = length(probs_plausible)
    sols = Array{Any}(undef, n_structures)
    for i in 1:n_structures
        plausible_prob = probs_plausible[i]
        callback_controls, initial_control, C = syms_cache[i]
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
        callback_controls, initial_control, C = syms_cache[i]
        for j in i+1:n_structures
            push!(squared_differences, maximum((sols[i][C] .- sols[j][C]) .^ 2))
        end
    end
    ret = -mean(squared_differences)
    println(ret)
    return ret
end


lb = zeros(15)          # min applied voltage
ub = 5.0 * ones(15)    # max applied voltage (V)
