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
@mtkmodel Bioreactor begin
    @constants begin
        C_s_in = 50.0
        y_x_s = 0.777
        m = 0.0
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
        σ ~ μ / y_x_s + m
        D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)
        D(C_x) ~ μ * C_x - Q_in / V * C_x
        D(V) ~ Q_in
    end
end
@mtkmodel TrueBioreactor begin
    @extend Bioreactor()
    @parameters begin
        μ_max = 0.421
        K_s = 0.439
        linear_control_slope = -0.1
        linear_control_intercept = 2.0
    end
    @equations begin
        Q_in ~ linear_control_intercept + linear_control_slope * t # this needs to be swapped to piecewise constant function
        μ ~ μ_max * C_s / (K_s + C_s) # this should be recovered from data
    end
end
@mtkbuild true_bioreactor = TrueBioreactor()
prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [])
sol = solve(prob, Tsit5())
plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
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
    @parameters begin
        linear_control_slope = -0.1, [tunable = false]
        linear_control_intercept = 2.0, [tunable = false]
    end
    @equations begin
        Q_in ~ linear_control_intercept + linear_control_slope * t # this needs to be swapped to piecewise constant function
        nn.output.u[1] ~ μ
        nn.input.u[1] ~ C_s
#=         μ ~ nn_out.u[1]
        nn_in.u[1] ~ C_s
        connect(nn_in, nn.input)
        connect(nn_out, nn.output) =#
    end
end

@mtkbuild  ude_bioreactor = UDEBioreactor()

ude_prob = ODEProblem{true, SciMLBase.FullSpecialize}(ude_bioreactor, [], (0.0, 15.0), [])
ude_sol = solve(ude_prob, Tsit5())

# we use all tunable parameters because it's easier on the remake
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))

get_vars = getu(ude_bioreactor, [ude_bioreactor.V, ude_bioreactor.C_s])
# TODO: Switch to data with noise instead of comparing against the reference sol
get_refs = getu(true_bioreactor, [true_bioreactor.V, true_bioreactor.C_s])

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
scatter!(sol, idxs=[true_bioreactor.V, true_bioreactor.C_s], ms=0.4, msw=0)
