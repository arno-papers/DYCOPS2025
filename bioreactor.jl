using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitNeuralNets
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEqTsit5, OrdinaryDiffEqNonlinearSolve, OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface
using Plots
using Optimization, OptimizationOptimisers
using SciMLStructures
using SciMLStructures: Tunable
using SciMLSensitivity
using SymbolicRegression
using LuxCore
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
prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, saveat = 0:15)
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

ude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, saveat = 0:15)
ude_sol = solve(ude_prob, Rodas5P())
plot(ude_sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

# we use all tunable parameters because it's easier on the remake
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))

get_vars = getu(ude_bioreactor, [ude_bioreactor.V, ude_bioreactor.C_s])
# TODO: Switch to data with noise instead of comparing against the reference sol
get_refs = getu(true_bioreactor, [true_bioreactor.V, true_bioreactor.C_s])

function loss(x, (prob, sol_ref, get_vars, get_refs))
    new_p = SciMLStructures.replace(Tunable(), prob.p, x)
    new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Rodas5P())
    loss = zero(eltype(x))
    for i in eachindex(new_sol.u)
        loss += sum(abs2.(get_vars(new_sol, i) .- get_refs(sol_ref, i)))
    end

    if SciMLBase.successful_retcode(new_sol)
        loss
    else
        Inf
    end

    loss
end

of = OptimizationFunction{true}(loss, AutoZygote())
ps = (ude_prob, sol, get_vars, get_refs);

op = OptimizationProblem(of, x0, ps)
of(x0, ps)[1]

plot_cb = (opt_state, loss,) -> begin
    @info "step $(opt_state.iter), loss: $loss"
    @info opt_state.u

    new_p = SciMLStructures.replace(Tunable(), ude_prob.p, opt_state.u)
    new_prob = remake(ude_prob, p = new_p)
    sol = solve(new_prob, Rodas5P())
    if SciMLBase.successful_retcode(sol)
        display(plot(sol))
    end
    false
end

res = solve(op, Optimization.LBFGS(), maxiters=100, callback=plot_cb)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())
plot(res_sol; label=["Cₛ(g/L) trained" "Cₓ(g/L) trained" "V(L) trained"], xlabel="t(h)", lw=3);
scatter!(sol; label=["Cₛ(g/L) true", "Cₓ(g/L) true", "V(L) true"], ms=3);
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600,legend=false)

# below this line no updates for the piecewise controls have been done yet.

## get chain from the equations
extracted_chain = arguments(equations(ude_sys.nn)[1].rhs)[1]
T = defaults(ude_sys)[ude_sys.nn.T]
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
plot(sol ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
for i in 1:length(model_structures)
    @mtkmodel plausible_bioreactor begin
        @parameters begin
            C_s_in = 50.0
            y_x_s = 0.777
            m = 0.0
            μ_max = 0.421
            K_s = 0.439
            linear_control_slope = -0.1
            linear_control_intercept = 2.0
        end
        @variables begin
            C_s(t) = 3.0
            C_x(t) = 0.25
            V(t) = 7.0
            Q_in(t)
            μ(t)
            σ(t)
        end
        @equations begin
            Q_in ~ linear_control_intercept + linear_control_slope * t # this needs to be swapped to piecewise constant function
            μ ~ model_structures[i](C_s)
            σ ~ μ / y_x_s + m
            D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
            D(C_x) ~ μ * C_x - Q_in / V * C_x
            D(V) ~ Q_in
        end
    end
    @mtkbuild plausible_bioreactor_f = plausible_bioreactor()
    prob_plausible = ODEProblem(plausible_bioreactor_f, [], (0.0, 15.0), [])
    sol_plausible  = solve(prob_plausible , Tsit5())
    plot!(sol_plausible ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

## simulate with different controls

## get complete plausible model structures
plot(sol ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
for i in 1:length(model_structures)
    @mtkmodel plausible_bioreactor begin
        @parameters begin
            C_s_in = 50.0
            y_x_s = 0.777
            m = 0.0
            μ_max = 0.421
            K_s = 0.439
            linear_control_slope = -0.1
            linear_control_intercept = 2.0
        end
        @variables begin
            C_s(t) = 3.0
            C_x(t) = 0.25
            V(t) = 7.0
            Q_in(t)
            μ(t)
            σ(t)
        end
        @equations begin
            Q_in ~ linear_control_intercept + linear_control_slope * t # this needs to be swapped to piecewise constant function
            μ ~ model_structures[i](C_s)
            σ ~ μ / y_x_s + m
            D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
            D(C_x) ~ μ * C_x - Q_in / V * C_x
            D(V) ~ Q_in
        end
    end
    @mtkbuild plausible_bioreactor_f = plausible_bioreactor()
    prob_plausible = ODEProblem(plausible_bioreactor_f, [], (0.0, 15.0), [])

    control_slope = plausible_bioreactor_f.linear_control_slope
    control_intercept = plausible_bioreactor_f.linear_control_intercept

    prob_plausible.ps[control_slope] = 2 * prob_plausible.ps[control_slope]
    prob_plausible.ps[control_intercept] = 2 * prob_plausible.ps[control_intercept]

    sol_plausible  = solve(prob_plausible , Tsit5())
    plot!(sol_plausible ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)
