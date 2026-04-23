using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("lotka_volterra_definitions.jl")

# first experiment

optimization_state =  zeros(15)
optimization_initial = optimization_state[1]
@mtkbuild true_lv = TrueLotkaVolterra()
prob = ODEProblem(true_lv, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = solve(prob, Rodas5P())

@mtkbuild ude_lv = UDELotkaVolterra()
ude_prob = ODEProblem(ude_lv, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = solve(ude_prob, Rodas5P())

data = DataFrame(sol)
data = data[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED

data[!, "N(t)"] += sd_N * randn(size(data, 1))

of = OptimizationFunction{true}(loss, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_lv),), tunable_parameters(ude_lv)))
get_vars = getu(ude_lv, [ude_lv.N])
ps = ([ude_prob], [get_vars], [data]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())

extracted_chain = arguments(equations(ude_lv.nn)[1].rhs)[1]
T = defaults(ude_lv)[ude_lv.nn.T]
N_range_plot = 0.0:0.01:10.0
ϕ_predicted_plot = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in N_range_plot]
ϕ_predicted_data = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in data[!, "N(t)"]]

## get plausible model structures for missing physics

hall_of_fame = equation_search(collect(data[!, "N(t)"])', ϕ_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[3], N_range_plot, model_structures[i].(N_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[1], sol_plausible, idxs=:N, lw=1,ls=:dash,c=i+2)
        plot!(plts[2], sol_plausible, idxs=:P, lw=1,ls=:dash,c=i+2)
    end
end
plot!(plts[1], sol, idxs=:N, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:N, lw=3,c=2)
plot!(plts[1], ylabel="N (prey)", xlabel="t")
scatter!(plts[1], data[!, "timestamp"], data[!, "N(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:P, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:P, lw=3,c=2)
plot!(plts[2], ylabel="P (predator)", xlabel="t")
plot!(plts[3], N_range_plot, ϕ_true.(N_range_plot), lw=3, c=1)
plot!(plts[3], N_range_plot, ϕ_predicted_plot, lw=3, c=2)
scatter!(plts[3], data[!, "N(t)"], ϕ_predicted_data, ms=3, c=2)
plot!(plts[3], ylabel="ϕ(N)", xlabel="N",ylims=(0,2.5))
plot(plts..., layout = 3, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("lv_experiment1.pdf")

# optimize the control pars
design_prob = OptimizationProblem(S_criterion, optimization_state, (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=100.0)

optimization_state = control_pars_opt.u
optimization_initial = optimization_initial2 = optimization_state[1]

plts = plot(), plot()
t_pwc = []
pwc = []
for i in 0:14
    push!(t_pwc,i)
    push!(t_pwc,i+1)
    push!(pwc,optimization_state[i+1])
    push!(pwc,optimization_state[i+1])
end
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t",ylabel="u (harvest rate)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, N = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[2], sol_plausible, idxs=:N, lw=3,ls=:dash,c=i+2)
    end
end
plot!(plts[2],xlabel="t",ylabel="N (prey)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("lv_control1.pdf")

# second experiment
@mtkbuild true_lv2 = TrueLotkaVolterra()
prob2 = ODEProblem(true_lv2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = solve(prob2, Rodas5P())
@mtkbuild ude_lv2 = UDELotkaVolterra()
ude_prob2 = ODEProblem(ude_lv2, [], (0.0, 15.0), [ude_lv2.u_harvest => optimization_initial], tstops=0:15, save_everystep=false)
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_prob_remake = remake(ude_prob, p=ude_prob2.p)
sol_remake = solve(ude_prob_remake, Rodas5P())
x0 = reduce(vcat, getindex.((default_values(ude_lv),), tunable_parameters(ude_lv)))

get_vars2 = getu(ude_lv2, [ude_lv2.N])

data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "N(t)"] += sd_N * randn(size(data2, 1))

ps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, NLopt.LN_BOBYQA, maxiters=5_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
callback_controls, initial_control, N = syms_cache[1]
res_prob.ps[initial_control] = optimization_initial2
res_sol = solve(res_prob, Rodas5P())
## get chain from the equations
extracted_chain = arguments(equations(ude_lv2.nn)[1].rhs)[1]
T = defaults(ude_lv2)[ude_lv2.nn.T]
ϕ_predicted_plot2 = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in N_range_plot]

ϕ_predicted_data = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in data[!, "N(t)"]]
ϕ_predicted_data2 = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in data2[!, "N(t)"]]

total_data = hcat(collect(data[!, "N(t)"]'), collect(data2[!, "N(t)"]'))
total_predicted_data = vcat(ϕ_predicted_data, ϕ_predicted_data2)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[3], N_range_plot, model_structures[i].(N_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[1], sol_plausible, idxs=:N, lw=1,ls=:dash,c=i+2)
        plot!(plts[2], sol_plausible, idxs=:P, lw=1,ls=:dash,c=i+2)
    end
end
plot!(plts[1], sol2, idxs=:N, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:N, lw=3,c=2)
plot!(plts[1], ylabel="N (prey)", xlabel="t")
scatter!(plts[1], data2[!, "timestamp"], data2[!, "N(t)"]; ms=3,c=1)
plot!(plts[2], sol2, idxs=:P, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:P, lw=3,c=2)
plot!(plts[2], ylabel="P (predator)", xlabel="t")
plot!(plts[3], N_range_plot, ϕ_true.(N_range_plot), lw=3, c=1)
plot!(plts[3], N_range_plot, ϕ_predicted_plot2, lw=3, c=2)
scatter!(plts[3], data[!, "N(t)"], ϕ_predicted_data, ms=3, c=2)
scatter!(plts[3], data2[!, "N(t)"], ϕ_predicted_data2, ms=3, c=2)
plot!(plts[3], ylabel="ϕ(N)", xlabel="N",ylims=(0,2.5))
plot(plts..., layout = 3, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("lv_experiment2.pdf")

S_criterion(zeros(15), (probs_plausible, syms_cache))

prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)

optimization_state = control_pars_opt.u
optimization_initial = optimization_state[1]

plts = plot(), plot()
t_pwc = []
pwc = []
for i in 0:14
    push!(t_pwc,i)
    push!(t_pwc,i+1)
    push!(pwc,optimization_state[i+1])
    push!(pwc,optimization_state[i+1])
end
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t",ylabel="u (harvest rate)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, N = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[2], sol_plausible, idxs=:N, lw=3,ls=:dash,c=i+2)
    end
end
plot!(plts[2],xlabel="t",ylabel="N (prey)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("lv_control2.pdf")


# third experiment
@mtkbuild true_lv3 = TrueLotkaVolterra()
prob3 = ODEProblem(true_lv3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol3 = solve(prob3, Rodas5P())
@mtkbuild ude_lv3 = UDELotkaVolterra()
ude_prob3 = ODEProblem(ude_lv3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)

x0 = reduce(vcat, getindex.((default_values(ude_lv3),), tunable_parameters(ude_lv3)))

get_vars3 = getu(ude_lv3, [ude_lv3.N])

data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "N(t)"] += sd_N * randn(size(data3, 1))

ps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=10_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob3.p, res.u)
res_prob = remake(ude_prob3, p=new_p)
res_sol = solve(res_prob, Rodas5P())

## get chain from the equations
extracted_chain = arguments(equations(ude_lv3.nn)[1].rhs)[1]
T = defaults(ude_lv3)[ude_lv3.nn.T]

ϕ_predicted_data = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in data[!, "N(t)"]]
ϕ_predicted_data2 = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in data2[!, "N(t)"]]
ϕ_predicted_data3 = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in data3[!, "N(t)"]]

total_data = hcat(collect(data[!, "N(t)"]'), collect(data2[!, "N(t)"]'), collect(data3[!, "N(t)"]'))
total_predicted_data = vcat(ϕ_predicted_data, ϕ_predicted_data2, ϕ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

scatter(total_data', total_predicted_data,legend=false)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plot()
ϕ_predicted_plot3 = [only(stateless_apply(extracted_chain, [N], convert(T,res.u))) for N in N_range_plot]
for i in 1:length(model_structures)
    plot!(N_range_plot, model_structures[i].(N_range_plot);c=i+2,lw=1,ls=:dash)
end
plot!(N_range_plot, ϕ_true.(N_range_plot), lw=3, c=1)
plot!(N_range_plot, ϕ_predicted_plot3, lw=3, c=2)
scatter!(data[!, "N(t)"], ϕ_predicted_data, ms=3, c=2)
scatter!(data2[!, "N(t)"], ϕ_predicted_data2, ms=3, c=2)
scatter!(data3[!, "N(t)"], ϕ_predicted_data3, ms=3, c=2)
plot!(ylabel="ϕ(N)", xlabel="N",ylims=(0,2.5),legend=false)
