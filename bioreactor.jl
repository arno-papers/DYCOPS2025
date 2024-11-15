using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("definitions.jl")

# first experiment

optimization_state =  zeros(15)
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor = TrueBioreactor()
prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = solve(prob, Rodas5P())

@mtkbuild  ude_bioreactor = UDEBioreactor()
ude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = solve(ude_prob, Rodas5P())

data1 = DataFrame(sol)
data1 = data1[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED; MTK ALWAYS SAVES BEFORE AND AFTER CALLBACK; WITH NO OPTION TO DISABLE

data1[!, "C_s(t)"] += sd_cs * randn(size(data1, 1))

function loss(x, (prob, sol_ref, get_vars, data))
    new_p = SciMLStructures.replace(Tunable(), prob.p, x)
    new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Rodas5P())
    loss = zero(eltype(x))
    for (i, j) in enumerate(1:2:length(new_sol.t))
        loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "C_s(t)"][i]))
    end
    loss
end

of = OptimizationFunction{true}(loss, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))
get_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])
ps = (ude_prob, sol, get_vars, data1);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())

extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
C_s_range_plot = 0.0:0.01:50.0
μ_predicted_plot = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]
μ_predicted_data = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]

## get plausible model structures for missing physics

hall_of_fame = equation_search(collect(data1[!, "C_s(t)"])', μ_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)
    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)
end
plot!(plts[1], sol, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], title="Cₛ(g/L)", xlabel="t(h)")
scatter!(data1[!, "timestamp"], data1[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], title="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol, idxs=:V, title="V(L)", xlabel="t(h)", lw=3, color=:black, ylims=(6.0,8.0))
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot, lw=3, c=2)
scatter!(plts[4], data1[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
plot!(plts[4], title="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
#Sebastian, swap title of subplots to x-axis label

# optimize the control pars
design_prob = OptimizationProblem(S_criterion, optimization_state, (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)

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
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="Qin (UNITS)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C_s = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)
end
plot!(plts[2],xlabel="t(h)",ylabel="Cₛ(g/L)")
plot(plts..., tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
#Sebastian make the above plot less elongated; also perhaps we will have to stack them vertically, depends on how the paper fills up

# second experiment
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = solve(prob2, Rodas5P())
@mtkbuild ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [ude_bioreactor2.Q_in => optimization_initial], tstops=0:15, save_everystep=false)

x0 = reduce(vcat, getindex.((default_values(ude_bioreactor2),), tunable_parameters(ude_bioreactor2))) # Sebastian; this should start from optimized pars from the first experiment.

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
    if !(SciMLBase.successful_retcode(new_sol1) & SciMLBase.successful_retcode(new_sol2))
        println("failed")
        return Inf
    end
    loss = zero(eltype(x))
    for (i, j) in enumerate(1:2:length(new_sol1.t))
        # @info "i: $i j: $j"
        loss += sum(abs2.(get_vars1(new_sol1, j) .- data1[!, "C_s(t)"][i]))
        loss += sum(abs2.(get_vars2(new_sol2, j) .- data2[!, "C_s(t)"][i]))
    end
    println(loss)
    loss
end
#Sebastian we really need a general loss function, such that we can fit dataset3

of = OptimizationFunction{true}(loss2, AutoZygote())
ps = (ude_prob, ude_prob2, get_vars, get_vars2, data1, data2);

op = OptimizationProblem(of, x0, ps)

res = solve(op, Optimization.LBFGS(), maxiters=1000)
new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
res_sol = solve(res_prob, Rodas5P())
## get chain from the equations
extracted_chain = arguments(equations(ude_bioreactor2.nn)[1].rhs)[1]
T = defaults(ude_bioreactor2)[ude_bioreactor2.nn.T]
μ_predicted_plot2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]

μ_predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]

total_data = hcat(collect(data1[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data1, μ_predicted_data2)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)
    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)
end
plot!(plts[1], sol2, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], title="Cₛ(g/L)", xlabel="t(h)")
scatter!(data2[!, "timestamp"], data2[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol2, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], title="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol2, idxs=:V, title="V(L)", xlabel="t(h)", lw=3, color=:black)
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot2, lw=3, c=2)
scatter!(plts[4], data1[!, "C_s(t)"], μ_predicted_data1, ms=3, c=2)
scatter!(plts[4], data2[!, "C_s(t)"], μ_predicted_data2, ms=3, c=2)
plot!(plts[4], title="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)


S_criterion(zeros(15), (probs_plausible, syms_cache))

prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)
