using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("electrochemical_definitions.jl")

# first experiment

optimization_state = 2.0 * ones(15) # moderate voltage for initial data (zero voltage = minimal dynamics)
optimization_initial = optimization_state[1]
@mtkbuild true_ec = TrueElectrochemicalCell()
prob = ODEProblem(true_ec, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = solve(prob, Rodas5P())

@mtkbuild ude_ec = UDEElectrochemicalCell()
ude_prob = ODEProblem(ude_ec, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = solve(ude_prob, Rodas5P())

data = DataFrame(sol)
data = data[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED

data[!, "C(t)"] += sd_C * randn(size(data, 1))

of = OptimizationFunction{true}(loss, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_ec),), tunable_parameters(ude_ec)))
get_vars = getu(ude_ec, [ude_ec.C])
ps = ([ude_prob], [get_vars], [data]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())

extracted_chain = arguments(equations(ude_ec.nn)[1].rhs)[1]
T = defaults(ude_ec)[ude_ec.nn.T]
eta_range_plot = -1.0:0.01:5.0
j_predicted_plot = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_range_plot]

# We need the overpotential values at measurement times to create SR training data
# Get eta from the trained UDE solution
new_p_res = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob_full = remake(ude_prob, p=new_p_res)
res_sol_full = solve(res_prob_full, Rodas5P())
get_eta = getu(ude_ec, [ude_ec.eta])
eta_data = [only(get_eta(res_sol_full, j)) for (i, j) in enumerate(1:2:length(res_sol_full.t))]
j_predicted_data = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_data]

## get plausible model structures for missing physics

hall_of_fame = equation_search(collect(eta_data)', j_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot()
for i in 1:length(model_structures)
    j_model = [try model_structures[i](eta) catch; NaN end for eta in eta_range_plot]
    plot!(plts[3], eta_range_plot, j_model;c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[1], sol_plausible, idxs=:C, lw=1,ls=:dash,c=i+2)
        plot!(plts[2], sol_plausible, idxs=:eta, lw=1,ls=:dash,c=i+2)
    end
end
plot!(plts[1], sol, idxs=:C, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C, lw=3,c=2)
plot!(plts[1], ylabel="C (concentration)", xlabel="t")
scatter!(plts[1], data[!, "timestamp"], data[!, "C(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:eta, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:eta, lw=3,c=2)
plot!(plts[2], ylabel="η (overpotential, V)", xlabel="t")
plot!(plts[3], eta_range_plot, j_true.(eta_range_plot), lw=3, c=1)
plot!(plts[3], eta_range_plot, j_predicted_plot, lw=3, c=2)
scatter!(plts[3], eta_data, j_predicted_data, ms=3, c=2)
plot!(plts[3], ylabel="j (current density)", xlabel="η (V)")
plot(plts..., layout = 3, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("ec_experiment1.pdf")

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
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t",ylabel="U (applied voltage, V)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[2], sol_plausible, idxs=:C, lw=3,ls=:dash,c=i+2)
    end
end
plot!(plts[2],xlabel="t",ylabel="C (concentration)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("ec_control1.pdf")

# second experiment
@mtkbuild true_ec2 = TrueElectrochemicalCell()
prob2 = ODEProblem(true_ec2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = solve(prob2, Rodas5P())
@mtkbuild ude_ec2 = UDEElectrochemicalCell()
ude_prob2 = ODEProblem(ude_ec2, [], (0.0, 15.0), [ude_ec2.U_applied => optimization_initial], tstops=0:15, save_everystep=false)
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_prob_remake = remake(ude_prob, p=ude_prob2.p)
sol_remake = solve(ude_prob_remake, Rodas5P())
x0 = reduce(vcat, getindex.((default_values(ude_ec),), tunable_parameters(ude_ec)))

get_vars2 = getu(ude_ec2, [ude_ec2.C])

data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C(t)"] += sd_C * randn(size(data2, 1))

ps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, NLopt.LN_BOBYQA, maxiters=5_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
callback_controls, initial_control, C = syms_cache[1]
res_prob.ps[initial_control] = optimization_initial2
res_sol = solve(res_prob, Rodas5P())
## get chain from the equations
extracted_chain = arguments(equations(ude_ec2.nn)[1].rhs)[1]
T = defaults(ude_ec2)[ude_ec2.nn.T]
j_predicted_plot2 = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_range_plot]

# Get eta values from trained UDE for both experiments
new_p_res1 = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob1 = remake(ude_prob, p=new_p_res1)
res_sol1 = solve(res_prob1, Rodas5P())
get_eta1 = getu(ude_ec, [ude_ec.eta])
eta_data = [only(get_eta1(res_sol1, j)) for (i, j) in enumerate(1:2:length(res_sol1.t))]

new_p_res2 = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob2_full = remake(ude_prob2, p=new_p_res2)
res_prob2_full.ps[initial_control] = optimization_initial2
res_sol2_full = solve(res_prob2_full, Rodas5P())
get_eta2 = getu(ude_ec2, [ude_ec2.eta])
eta_data2 = [only(get_eta2(res_sol2_full, j)) for (i, j) in enumerate(1:2:length(res_sol2_full.t))]

j_predicted_data = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_data]
j_predicted_data2 = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_data2]

total_eta = hcat(collect(eta_data)', collect(eta_data2)')
total_predicted_j = vcat(j_predicted_data, j_predicted_data2)
hall_of_fame = equation_search(total_eta, total_predicted_j; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot()
for i in 1:length(model_structures)
    j_model = [try model_structures[i](eta) catch; NaN end for eta in eta_range_plot]
    plot!(plts[3], eta_range_plot, j_model;c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[1], sol_plausible, idxs=:C, lw=1,ls=:dash,c=i+2)
        plot!(plts[2], sol_plausible, idxs=:eta, lw=1,ls=:dash,c=i+2)
    end
end
plot!(plts[1], sol2, idxs=:C, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C, lw=3,c=2)
plot!(plts[1], ylabel="C (concentration)", xlabel="t")
scatter!(plts[1], data2[!, "timestamp"], data2[!, "C(t)"]; ms=3,c=1)
plot!(plts[2], sol2, idxs=:eta, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:eta, lw=3,c=2)
plot!(plts[2], ylabel="η (overpotential, V)", xlabel="t")
plot!(plts[3], eta_range_plot, j_true.(eta_range_plot), lw=3, c=1)
plot!(plts[3], eta_range_plot, j_predicted_plot2, lw=3, c=2)
scatter!(plts[3], eta_data, j_predicted_data, ms=3, c=2)
scatter!(plts[3], eta_data2, j_predicted_data2, ms=3, c=2)
plot!(plts[3], ylabel="j (current density)", xlabel="η (V)")
plot(plts..., layout = 3, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("ec_experiment2.pdf")

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
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t",ylabel="U (applied voltage, V)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol_plausible)
        plot!(plts[2], sol_plausible, idxs=:C, lw=3,ls=:dash,c=i+2)
    end
end
plot!(plts[2],xlabel="t",ylabel="C (concentration)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("ec_control2.pdf")


# third experiment
@mtkbuild true_ec3 = TrueElectrochemicalCell()
prob3 = ODEProblem(true_ec3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol3 = solve(prob3, Rodas5P())
@mtkbuild ude_ec3 = UDEElectrochemicalCell()
ude_prob3 = ODEProblem(ude_ec3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)

x0 = reduce(vcat, getindex.((default_values(ude_ec3),), tunable_parameters(ude_ec3)))

get_vars3 = getu(ude_ec3, [ude_ec3.C])

data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C(t)"] += sd_C * randn(size(data3, 1))

ps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=10_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob3.p, res.u)
res_prob = remake(ude_prob3, p=new_p)
res_sol = solve(res_prob, Rodas5P())

## get chain from the equations
extracted_chain = arguments(equations(ude_ec3.nn)[1].rhs)[1]
T = defaults(ude_ec3)[ude_ec3.nn.T]

# Get eta from all three experiments
new_p_res1 = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob1 = remake(ude_prob, p=new_p_res1)
res_sol1 = solve(res_prob1, Rodas5P())
eta_data = [only(get_eta1(res_sol1, j)) for (i, j) in enumerate(1:2:length(res_sol1.t))]

new_p_res2 = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob2_full = remake(ude_prob2, p=new_p_res2)
res_prob2_full.ps[initial_control] = optimization_initial2
res_sol2_full = solve(res_prob2_full, Rodas5P())
eta_data2 = [only(get_eta2(res_sol2_full, j)) for (i, j) in enumerate(1:2:length(res_sol2_full.t))]

get_eta3 = getu(ude_ec3, [ude_ec3.eta])
new_p_res3 = SciMLStructures.replace(Tunable(), ude_prob3.p, res.u)
res_prob3_full = remake(ude_prob3, p=new_p_res3)
res_sol3_full = solve(res_prob3_full, Rodas5P())
eta_data3 = [only(get_eta3(res_sol3_full, j)) for (i, j) in enumerate(1:2:length(res_sol3_full.t))]

j_predicted_data = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_data]
j_predicted_data2 = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_data2]
j_predicted_data3 = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_data3]

total_eta = hcat(collect(eta_data)', collect(eta_data2)', collect(eta_data3)')
total_predicted_j = vcat(j_predicted_data, j_predicted_data2, j_predicted_data3)
hall_of_fame = equation_search(total_eta, total_predicted_j; options, niterations=1000, runtests=false, parallelism=:serial)

scatter(total_eta', total_predicted_j,legend=false)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plot()
j_predicted_plot3 = [only(stateless_apply(extracted_chain, [eta], convert(T,res.u))) for eta in eta_range_plot]
for i in 1:length(model_structures)
    j_model = [try model_structures[i](eta) catch; NaN end for eta in eta_range_plot]
    plot!(eta_range_plot, j_model;c=i+2,lw=1,ls=:dash)
end
plot!(eta_range_plot, j_true.(eta_range_plot), lw=3, c=1)
plot!(eta_range_plot, j_predicted_plot3, lw=3, c=2)
scatter!(eta_data, j_predicted_data, ms=3, c=2)
scatter!(eta_data2, j_predicted_data2, ms=3, c=2)
scatter!(eta_data3, j_predicted_data3, ms=3, c=2)
plot!(ylabel="j (current density)", xlabel="η (V)",legend=false)
