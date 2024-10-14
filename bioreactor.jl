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

@mtkmodel true_bioreactor begin
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
        μ ~ μ_max * C_s / (K_s + C_s) # this should be recovered from data
        σ ~ μ / y_x_s + m
        D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
        D(C_x) ~ μ * C_x - Q_in / V * C_x
        D(V) ~ Q_in
    end
end
@mtkbuild true_bioreactor_f = true_bioreactor()
prob = ODEProblem(true_bioreactor_f, [], (0.0, 15.0), [])
sol = solve(prob, Tsit5())
plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

function incomplete_bioreactor(; name)
    @parameters begin
        C_s_in = 50.0, [tunable = false]
        y_x_s = 0.777, [tunable = false]
        m = 0.0, [tunable = false]
        μ_max = 0.421, [tunable = false]
        K_s = 0.439, [tunable = false]
        linear_control_slope = -0.1, [tunable = false]
        linear_control_intercept = 2.0, [tunable = false]
    end
    @variables begin
        C_s(t) = 3.0
        C_x(t) = 0.25
        V(t) = 7.0
        Q_in(t)
        μ(t)
        σ(t)
    end

    @named nn_in = RealInputArray(nin=1)
    @named nn_out = RealOutputArray(nout=1)

    eqs = [
        Q_in ~ linear_control_intercept + linear_control_slope * t # this needs to be swapped to piecewise constant function
        μ ~ nn_out.u[1]
        σ ~ μ / y_x_s + m
        D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
        D(C_x) ~ μ * C_x - Q_in / V * C_x
        D(V) ~ Q_in
        nn_in.u[1] ~ C_s
    ]

    return ODESystem(eqs, t; systems = [nn_in, nn_out], name)
end

function bioreactor_UDE(; name)
    chain = multi_layer_feed_forward(1, 1)

    @named bioreactor = incomplete_bioreactor()
    @named nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain)

    eqs = [
        connect(bioreactor.nn_in, nn.input)
        connect(bioreactor.nn_out, nn.output)
    ]
    # note that defaults are (currently) required for the array variable due to
    # issues in initialization
    ODESystem(eqs, t; defaults=[nn.input.u => [0.0]], systems=[bioreactor, nn], name)
end

ude_sys = structural_simplify(bioreactor_UDE(name=:ude_sys))

ude_prob = ODEProblem{true, SciMLBase.FullSpecialize}(ude_sys, [], (0.0, 15.0), [])
ude_sol = solve(ude_prob, Rodas5P())

# we use all tunable parameters because it's easier on the remake
x0 = reduce(vcat, getindex.((default_values(ude_sys),), tunable_parameters(ude_sys)))

get_vars = getu(ude_sys, [ude_sys.bioreactor.V, ude_sys.bioreactor.C_s])
# TODO: Switch to data with noise instead of comparing against the reference sol
get_refs = getu(true_bioreactor_f, [true_bioreactor_f.V, true_bioreactor_f.C_s])

function loss(x, (prob, sol_ref, get_vars, get_refs))
    new_p = SciMLStructures.replace(Tunable(), prob.p, x)
    new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Rodas5P(), saveat=ts)

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
plot(res_sol)
scatter!(sol, idxs=[true_bioreactor_f.V, true_bioreactor_f.C_s], ms=0.4, msw=0)

# continued by Arno

using LuxCore

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
using SymbolicRegression
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
    prob_plausible.p[1][2] = 2*prob_plausible.p[1][2]
    prob_plausible.p[1][3] = 2*prob_plausible.p[1][3] # HARDCODED control par indices (not sure if the index is garanteed to always be 2 and 3)
    sol_plausible  = solve(prob_plausible , Tsit5())
    plot!(sol_plausible ; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
end
plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)
