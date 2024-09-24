using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5, OrdinaryDiffEqNonlinearSolve, OrdinaryDiffEqRosenbrock
using Plots
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

