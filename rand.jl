function create_bioreactor(optimization_state)
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
            C_s(t) = 1.0
            C_x(t) = 1.0
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
end

function create_true_bioreactor(optimization_state)
    Bioreactor = create_bioreactor(optimization_state)
    @mtkmodel TrueBioreactor begin
        @extend Bioreactor()
        @parameters begin
            μ_max = 0.421
            K_s = 0.439 * 10
        end
        @equations begin
            μ ~ μ_max * C_s / (K_s + C_s) # this should be recovered from data
        end
    end
    @mtkbuild true_bioreactor = TrueBioreactor()
end

function create_ude_bioreactor(optimization_state, rng)
    Bioreactor = create_bioreactor(optimization_state)

    @mtkmodel UDEBioreactor begin
        @extend Bioreactor()
        @structural_parameters begin
            chain = Lux.Chain(Lux.Dense(1, 5, tanh),
                Lux.Dense(5, 5, tanh),
                Lux.Dense(5, 1, x -> 1 * sigmoid(x)))
        end
        @components begin
            #=         nn_in = RealInputArray(nin=1)
                    nn_out = RealOutputArray(nout=1) =#
            nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)
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

    @mtkbuild ude_bioreactor = UDEBioreactor()
end

function loss(x, (prob, sol_ref, get_vars, data))
    new_p = SciMLStructures.replace(Tunable(), prob.p, x)
    new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))
    ts = sol_ref.t
    new_sol = solve(new_prob, Rodas5P())
    loss = zero(eltype(x))
    for (i, j) in enumerate(1:2:length(new_sol.t))
        # @info "i: $i j: $j"
        loss += sum(abs2.(get_vars(new_sol, j) .- data[!, "C_s(t)"][i]))
    end

    if SciMLBase.successful_retcode(new_sol)
        loss
    else
        Inf
    end

    loss
end

function bioreactor_UDE_plausible_models(optimization_state; rng)
    # optimization_state = zeros(15)

    true_bioreactor = create_true_bioreactor(optimization_state)

    prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
    sol = solve(prob, Rodas5P())
    plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

    ude_bioreactor = create_ude_bioreactor(optimization_state, rng)

    ude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
    ude_sol = solve(ude_prob, Rodas5P())
    plot(ude_sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600)

    # we use all tunable parameters because it's easier on the remake
    x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))

    get_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])
    # TODO: Switch to data with noise instead of comparing against the reference sol
    data = DataFrame(sol)
    data = data[1:2:end, :]

    sd_cs = 0.1
    data[!, "C_s(t)"] += sd_cs * randn(size(data, 1))

    plt = plot(sol)
    scatter!(data[!, "timestamp"], data[!, "C_s(t)"]; label="Cₛ(g/L) true", ms=3)
    scatter!(data[!, "timestamp"], data[!, "C_x(t)"]; label="Cₓ(g/L) true", ms=3)
    scatter!(data[!, "timestamp"], data[!, "V(t)"]; label="V(L) true", ms=3)

    display(plt)

    # get_refs = getu(true_bioreactor, [true_bioreactor.V, true_bioreactor.C_s])

    of = OptimizationFunction{true}(loss, AutoZygote())
    ps = (ude_prob, sol, get_vars, data)

    op = OptimizationProblem(of, x0, ps)
    of(x0, ps)[1]

    plot_cb = (opt_state, loss,) -> begin
        @info "step $(opt_state.iter), loss: $loss"
        # @info opt_state.u

        new_p = SciMLStructures.replace(Tunable(), ude_prob.p, opt_state.u)
        new_prob = remake(ude_prob, p=new_p)
        sol = solve(new_prob, Rodas5P())
        if SciMLBase.successful_retcode(sol)
            display(plot(sol))
        end
        false
    end

    @info "Starting optimization"

    res = solve(op, Optimization.LBFGS(), maxiters=1000)

    new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
    res_prob = remake(ude_prob, p=new_p)
    res_sol = solve(res_prob, Rodas5P())
    plt = plot(res_sol; label=["Cₛ(g/L) trained" "Cₓ(g/L) trained" "V(L) trained"], xlabel="t(h)", lw=3)
    scatter!(data[!, "timestamp"], data[!, "C_s(t)"]; label=["Cₛ(g/L) true",], ms=3)
    scatter!(data[!, "timestamp"], data[!, "C_x(t)"]; label=["Cₓ(g/L) true",], ms=3)
    scatter!(data[!, "timestamp"], data[!, "V(t)"]; label=["V(L) true"], ms=3)

    display(plt)

    ## get chain from the equations
    extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
    T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
    C_s = LuxCore.stateless_apply(extracted_chain, [20.0], convert(T, res.u))
    C_s_range = range(minimum(data[!, "C_s(t)"]), maximum(data[!, "C_s(t)"]), 100)
    C_s_range_plot = 0.0:0.01:50.0
    C_s_train =
        μ_predicted = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T, res.u))) for C_s in C_s_range]
    μ_predicted_plot = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T, res.u))) for C_s in C_s_range_plot]

    μ_max = 0.421
    K_s = 0.439 * 10
    plt = plot(C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot))
    plot!(C_s_range_plot, μ_predicted_plot)
    predicted_data = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T, res.u))) for C_s in data[!, "C_s(t)"]]
    scatter!(data[!, "C_s(t)"], predicted_data)

    @info "Starting equation search"

    ## get plausible model structures for missing physics
    options = SymbolicRegression.Options(
        unary_operators=(exp, sin, cos),
        binary_operators=(+, *, /, -),
        seed=123,
        deterministic=true,
        save_to_file=false
    )
    hall_of_fame = equation_search(collect(data[!, "C_s(t)"])', predicted_data; options, niterations=100, runtests=false, parallelism=:serial)

    n_best_max = 10
    n_best = min(length(hall_of_fame.members), n_best_max) #incase < 10 model structures were returned
    best_models = []
    best_models_scores = []
    i = 1
    round(hall_of_fame.members[i].loss, sigdigits=5)
    while length(best_models) <= n_best
        println(i)
        member = hall_of_fame.members[i]
        rounded_score = round(member.loss, sigdigits=5)
        if !(rounded_score in best_models_scores)
            push!(best_models, member)
            push!(best_models_scores, rounded_score)
        end
        i += 1
    end
    best_models

    @syms x
    eqn = node_to_symbolic(best_models[1].tree, options, variable_names=["x"])

    f = build_function(eqn, x, expression=Val{false})
    f.(collect(C_s_range)')

    function get_model_structures(best_models, options, n_best, plt)
        model_structures = []
        @syms x

        for i = 1:n_best
            eqn = node_to_symbolic(best_models[i].tree, options, varMap=["x"])
            fi = build_function(eqn, x, expression=Val{false})
            x_plot = Float64[]
            y_plot = Float64[]
            for x_try in C_s_range_plot
                try
                    y_try = fi(x_try)
                    append!(x_plot, x_try)
                    append!(y_plot, y_try)
                catch
                end
            end
            plot!(plt, x_plot, y_plot, label="model $i")
            push!(model_structures, fi)
        end
        plot!(plt, legend=false, ylims=(0, 1))

        return model_structures
    end

    model_structures = get_model_structures(best_models, options, n_best, plt)
    plt

    ## get complete plausible model structures
    plot(sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    for i in 1:length(model_structures)
        @mtkmodel PlausibleBioreactor begin
            @extend Bioreactor()
            @equations begin
                μ ~ model_structures[i](C_s)
            end
        end
        @mtkbuild plausible_bioreactor = PlausibleBioreactor()
        plausible_prob = ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)
        plausible_sol = solve(plausible_prob, Rodas5P())
        plot!(plausible_sol; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    end
    plot!(tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, dpi=600, legend=false)
end

optimization_state = zeros(15)

bioreactor_UDE_plausible_models(optimization_state; rng)

bioreactor_UDE_plausible_models(rand(15); rng)
