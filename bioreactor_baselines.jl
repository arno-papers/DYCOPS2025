using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("definitions.jl")

# ============================================================================
# Baseline comparison for the bioreactor example
# Compares: S-criterion (M=10), T-optimal (M=2), uniform grid, random
# ============================================================================

# T-optimal criterion: discriminate between 2 specific model structures
# (best and second-best from SR after first experiment)
function T_criterion(optimization_state, (probs_plausible, syms_cache))
    # Use only the first 2 models (best and second-best)
    n_structures = 2
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
    callback_controls, initial_control, C_s = syms_cache[1]
    ret = -maximum((sols[1][C_s] .- sols[2][C_s]) .^ 2)
    println(ret)
    return ret
end

# ============================================================================
# Helper: run the full 3-experiment workflow with a given control strategy
# ============================================================================
function run_baseline(control_strategy::Symbol; n_runs=5)
    results = []
    for run_id in 1:n_runs
        println("\n===== $(control_strategy) run $(run_id) =====")

        # --- First experiment: always zero control ---
        optimization_state = zeros(15)
        optimization_initial = optimization_state[1]
        @mtkbuild true_bioreactor = TrueBioreactor()
        prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
        sol = solve(prob, Rodas5P())

        @mtkbuild ude_bioreactor = UDEBioreactor()
        ude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

        data = DataFrame(sol)
        data = data[1:2:end, :]
        data[!, "C_s(t)"] += sd_cs * randn(size(data, 1))

        of = OptimizationFunction{true}(loss, AutoZygote())
        x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))
        get_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])
        ps = ([ude_prob], [get_vars], [data])
        op = OptimizationProblem(of, x0, ps)
        res = solve(op, Optimization.LBFGS(), maxiters=1000)

        extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
        T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
        μ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
        hall_of_fame = equation_search(collect(data[!, "C_s(t)"])', μ_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)
        model_structures = get_model_structures(hall_of_fame, options, n_best)
        probs_plausible, syms_cache = get_probs_and_caches(model_structures)

        # --- Second experiment: apply control strategy ---
        if control_strategy == :random
            optimization_state = rand(15) * 10
        elseif control_strategy == :uniform_grid
            optimization_state = collect(range(0.0, 10.0, length=15))
        elseif control_strategy == :t_optimal
            # T-optimal: optimize to discriminate between best 2 models
            design_prob = OptimizationProblem(T_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
            control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=100.0)
            optimization_state = control_pars_opt.u
        elseif control_strategy == :s_optimal
            # S-criterion (M=10): full method
            design_prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
            control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=100.0)
            optimization_state = control_pars_opt.u
        end
        optimization_initial = optimization_initial2 = optimization_state[1]

        @mtkbuild true_bioreactor2 = TrueBioreactor()
        prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
        sol2 = solve(prob2, Rodas5P())
        @mtkbuild ude_bioreactor2 = UDEBioreactor()
        ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [ude_bioreactor2.Q_in => optimization_initial], tstops=0:15, save_everystep=false)

        get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])
        data2 = DataFrame(sol2)
        data2 = data2[1:2:end, :]
        data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))

        ps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2])
        op = OptimizationProblem(of, x0, ps)
        res = solve(op, NLopt.LN_BOBYQA, maxiters=5_000)

        extracted_chain = arguments(equations(ude_bioreactor2.nn)[1].rhs)[1]
        T = defaults(ude_bioreactor2)[ude_bioreactor2.nn.T]
        μ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
        μ_predicted_data2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
        total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'))
        total_predicted_data = vcat(μ_predicted_data, μ_predicted_data2)
        hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)
        model_structures = get_model_structures(hall_of_fame, options, n_best)
        probs_plausible, syms_cache = get_probs_and_caches(model_structures)

        # --- Third experiment: apply control strategy again ---
        if control_strategy == :random
            optimization_state = rand(15) * 10
        elseif control_strategy == :uniform_grid
            optimization_state = collect(range(10.0, 0.0, length=15))  # reverse direction
        elseif control_strategy == :t_optimal
            design_prob = OptimizationProblem(T_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
            control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)
            optimization_state = control_pars_opt.u
        elseif control_strategy == :s_optimal
            design_prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
            control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)
            optimization_state = control_pars_opt.u
        end
        optimization_initial = optimization_state[1]

        @mtkbuild true_bioreactor3 = TrueBioreactor()
        prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
        sol3 = solve(prob3, Rodas5P())
        @mtkbuild ude_bioreactor3 = UDEBioreactor()
        ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)

        x0 = reduce(vcat, getindex.((default_values(ude_bioreactor3),), tunable_parameters(ude_bioreactor3)))
        get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])
        data3 = DataFrame(sol3)
        data3 = data3[1:2:end, :]
        data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))

        ps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3])
        op = OptimizationProblem(of, x0, ps)
        res = solve(op, Optimization.LBFGS(), maxiters=10_000)

        extracted_chain = arguments(equations(ude_bioreactor3.nn)[1].rhs)[1]
        T = defaults(ude_bioreactor3)[ude_bioreactor3.nn.T]
        μ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
        μ_predicted_data2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
        μ_predicted_data3 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]
        total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
        total_predicted_data = vcat(μ_predicted_data, μ_predicted_data2, μ_predicted_data3)
        hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

        # Report results
        println("\n--- $(control_strategy) run $(run_id) final hall of fame ---")
        for member in hall_of_fame.members[1:min(end, 10)]
            @syms x
            eqn = node_to_symbolic(member.tree, options, varMap=["x"])
            println("  loss=$(round(member.loss, sigdigits=5)): $eqn")
        end
        push!(results, hall_of_fame)
    end
    return results
end

# ============================================================================
# Run all baselines
# ============================================================================
println("\n" * "="^60)
println("RANDOM BASELINE")
println("="^60)
random_results = run_baseline(:random, n_runs=5)

println("\n" * "="^60)
println("UNIFORM GRID BASELINE")
println("="^60)
uniform_results = run_baseline(:uniform_grid, n_runs=5)

println("\n" * "="^60)
println("T-OPTIMAL (M=2) BASELINE")
println("="^60)
t_optimal_results = run_baseline(:t_optimal, n_runs=5)

println("\n" * "="^60)
println("S-OPTIMAL (M=10) - PROPOSED METHOD")
println("="^60)
s_optimal_results = run_baseline(:s_optimal, n_runs=5)
