using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("distillation_definitions.jl")

# first experiment

optimization_state = 2.0 * ones(15) # moderate boilup for initial data gathering (zero boilup = no dynamics)
optimization_initial = optimization_state[1]
@mtkbuild true_dist = TrueBatchDistillation()
prob = ODEProblem(true_dist, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = solve(prob, Rodas5P())

@mtkbuild ude_dist = UDEBatchDistillation()
ude_prob = ODEProblem(ude_dist, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = solve(ude_prob, Rodas5P())

data = DataFrame(sol)
data = data[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED

data[!, "x(t)"] += sd_x * randn(size(data, 1))

of = OptimizationFunction{true}(loss, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_dist),), tunable_parameters(ude_dist)))
get_vars = getu(ude_dist, [ude_dist.x])
ps = ([ude_prob], [get_vars], [data]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())

extracted_chain = arguments(equations(ude_dist.nn)[1].rhs)[1]
T = defaults(ude_dist)[ude_dist.nn.T]
x_range_plot = 0.0:0.005:1.0
y_predicted_plot = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in x_range_plot]
y_predicted_data = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in data[!, "x(t)"]]

## get plausible model structures for missing physics

hall_of_fame = equation_search(collect(data[!, "x(t)"])', y_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot()
for i in 1:length(model_structures)
    # clamp model structure output to [0,1] for plotting
    y_model = clamp.(model_structures[i].(x_range_plot), 0.0, 1.0)
    plot!(plts[3], x_range_plot, y_model;c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[1], sol_plausible, idxs=:x, lw=1,ls=:dash,c=i+2)
        plot!(plts[2], sol_plausible, idxs=:L, lw=1,ls=:dash,c=i+2)
    end
end
plot!(plts[1], sol, idxs=:x, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:x, lw=3,c=2)
plot!(plts[1], ylabel="x (liquid comp.)", xlabel="t(h)")
scatter!(plts[1], data[!, "timestamp"], data[!, "x(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:L, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:L, lw=3,c=2)
plot!(plts[2], ylabel="L (holdup, mol)", xlabel="t(h)")
plot!(plts[3], x_range_plot, y_vle.(x_range_plot), lw=3, c=1)
plot!(plts[3], x_range_plot, y_predicted_plot, lw=3, c=2)
scatter!(plts[3], data[!, "x(t)"], y_predicted_data, ms=3, c=2)
plot!(plts[3], ylabel="y (vapor comp.)", xlabel="x (liquid comp.)",ylims=(0,1))
# add diagonal reference line y=x
plot!(plts[3], x_range_plot, x_range_plot, lw=1, c=:gray, ls=:dot)
plot(plts..., layout = 3, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("dist_experiment1.pdf")

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
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="V (boilup, mol/h)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, x_var = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[2], sol_plausible, idxs=:x, lw=3,ls=:dash,c=i+2)
    end
end
plot!(plts[2],xlabel="t(h)",ylabel="x (liquid comp.)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("dist_control1.pdf")

# second experiment
@mtkbuild true_dist2 = TrueBatchDistillation()
prob2 = ODEProblem(true_dist2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = solve(prob2, Rodas5P())
@mtkbuild ude_dist2 = UDEBatchDistillation()
ude_prob2 = ODEProblem(ude_dist2, [], (0.0, 15.0), [ude_dist2.V_rate => optimization_initial], tstops=0:15, save_everystep=false)
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_prob_remake = remake(ude_prob, p=ude_prob2.p)
sol_remake = solve(ude_prob_remake, Rodas5P())
x0 = reduce(vcat, getindex.((default_values(ude_dist),), tunable_parameters(ude_dist)))

get_vars2 = getu(ude_dist2, [ude_dist2.x])

data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "x(t)"] += sd_x * randn(size(data2, 1))

ps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, NLopt.LN_BOBYQA, maxiters=5_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
callback_controls, initial_control, x_var = syms_cache[1]
res_prob.ps[initial_control] = optimization_initial2
res_sol = solve(res_prob, Rodas5P())
## get chain from the equations
extracted_chain = arguments(equations(ude_dist2.nn)[1].rhs)[1]
T = defaults(ude_dist2)[ude_dist2.nn.T]
y_predicted_plot2 = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in x_range_plot]

y_predicted_data = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in data[!, "x(t)"]]
y_predicted_data2 = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in data2[!, "x(t)"]]

total_data = hcat(collect(data[!, "x(t)"]'), collect(data2[!, "x(t)"]'))
total_predicted_data = vcat(y_predicted_data, y_predicted_data2)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot()
for i in 1:length(model_structures)
    y_model = clamp.(model_structures[i].(x_range_plot), 0.0, 1.0)
    plot!(plts[3], x_range_plot, y_model;c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[1], sol_plausible, idxs=:x, lw=1,ls=:dash,c=i+2)
        plot!(plts[2], sol_plausible, idxs=:L, lw=1,ls=:dash,c=i+2)
    end
end
plot!(plts[1], sol2, idxs=:x, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:x, lw=3,c=2)
plot!(plts[1], ylabel="x (liquid comp.)", xlabel="t(h)")
scatter!(plts[1], data2[!, "timestamp"], data2[!, "x(t)"]; ms=3,c=1)
plot!(plts[2], sol2, idxs=:L, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:L, lw=3,c=2)
plot!(plts[2], ylabel="L (holdup, mol)", xlabel="t(h)")
plot!(plts[3], x_range_plot, y_vle.(x_range_plot), lw=3, c=1)
plot!(plts[3], x_range_plot, y_predicted_plot2, lw=3, c=2)
scatter!(plts[3], data[!, "x(t)"], y_predicted_data, ms=3, c=2)
scatter!(plts[3], data2[!, "x(t)"], y_predicted_data2, ms=3, c=2)
plot!(plts[3], ylabel="y (vapor comp.)", xlabel="x (liquid comp.)",ylims=(0,1))
plot!(plts[3], x_range_plot, x_range_plot, lw=1, c=:gray, ls=:dot)
plot(plts..., layout = 3, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("dist_experiment2.pdf")

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
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="V (boilup, mol/h)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, x_var = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[2], sol_plausible, idxs=:x, lw=3,ls=:dash,c=i+2)
    end
end
plot!(plts[2],xlabel="t(h)",ylabel="x (liquid comp.)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("dist_control2.pdf")


# third experiment
@mtkbuild true_dist3 = TrueBatchDistillation()
prob3 = ODEProblem(true_dist3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol3 = solve(prob3, Rodas5P())
@mtkbuild ude_dist3 = UDEBatchDistillation()
ude_prob3 = ODEProblem(ude_dist3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)

x0 = reduce(vcat, getindex.((default_values(ude_dist3),), tunable_parameters(ude_dist3)))

get_vars3 = getu(ude_dist3, [ude_dist3.x])

data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "x(t)"] += sd_x * randn(size(data3, 1))

ps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=10_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob3.p, res.u)
res_prob = remake(ude_prob3, p=new_p)
res_sol = solve(res_prob, Rodas5P())

## get chain from the equations
extracted_chain = arguments(equations(ude_dist3.nn)[1].rhs)[1]
T = defaults(ude_dist3)[ude_dist3.nn.T]

y_predicted_data = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in data[!, "x(t)"]]
y_predicted_data2 = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in data2[!, "x(t)"]]
y_predicted_data3 = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in data3[!, "x(t)"]]

total_data = hcat(collect(data[!, "x(t)"]'), collect(data2[!, "x(t)"]'), collect(data3[!, "x(t)"]'))
total_predicted_data = vcat(y_predicted_data, y_predicted_data2, y_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

scatter(total_data', total_predicted_data,legend=false)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plot()
y_predicted_plot3 = [only(stateless_apply(extracted_chain, [xv], convert(T,res.u))) for xv in x_range_plot]
for i in 1:length(model_structures)
    y_model = clamp.(model_structures[i].(x_range_plot), 0.0, 1.0)
    plot!(x_range_plot, y_model;c=i+2,lw=1,ls=:dash)
end
plot!(x_range_plot, y_vle.(x_range_plot), lw=3, c=1)
plot!(x_range_plot, y_predicted_plot3, lw=3, c=2)
scatter!(data[!, "x(t)"], y_predicted_data, ms=3, c=2)
scatter!(data2[!, "x(t)"], y_predicted_data2, ms=3, c=2)
scatter!(data3[!, "x(t)"], y_predicted_data3, ms=3, c=2)
plot!(x_range_plot, x_range_plot, lw=1, c=:gray, ls=:dot)
plot!(ylabel="y (vapor comp.)", xlabel="x (liquid comp.)",ylims=(0,1),legend=false)
