var documenterSearchIndex = {"docs":
[{"location":"Optimal Data Gathering for Missing Physics/#Optimal-Data-Gathering-for-Missing-Physics.","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"","category":"section"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The missing physics showcase teaches how to discover the missing parts of a dynamic model, using universal differential equations (UDE) and symbolic regression (SR).","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"High quality data is needed to ensure the true dynamics are recovered. In this tutorial, we look at an efficient data gathering technique for SciML models, using a bioreactor example. To this end, we will rely on the following packages:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"using Random; Random.seed!(984519674645)\nusing StableRNGs; rng = StableRNG(845652695)\nusing ModelingToolkit\nusing ModelingToolkit: t_nounits as t, D_nounits as D\nusing ModelingToolkitNeuralNets\nusing OrdinaryDiffEqRosenbrock\nusing SymbolicIndexingInterface\nusing Plots\nusing Optimization, OptimizationOptimisers, OptimizationBBO, OptimizationNLopt\nusing SciMLStructures\nusing SciMLStructures: Tunable\nusing SciMLSensitivity\nusing Statistics\nusing SymbolicRegression\nusing LuxCore\nusing LuxCore: stateless_apply\nusing Lux\nusing Statistics\nusing DataFrames\nnothing # hide","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The bioreactor consists of 3 states: substrate concentration C_s(t), biomass concentration C_x(t) and volume V(t).","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"beginaligned\nfracdC_sdt = -left(fracmu(C_s)y_xs + mright) C_x + fracQ_in(t)V(C_Sin - C_s)\nfracdC_xdt = mu(C_s) C_x - fracQ_in(t)VC_x\nfracdVdt = Q_in(t)\nendaligned","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The substrate is eaten by the biomass, causing the biomass to grow. The rate by which the biomass grows μ(t) is an unknown function (missing physics), which must be estimated from experimental data. The rate by which the substrate is consumed σ(t) is dependent on μ(t), trough a yield factor y_xs and a maintenance term m, where are assumed to be known parameters. More substrate can be pumped into the reactor  with pumping speed Q_in(t). This pumped substrate has known concentration C_s_in. The goal is to optimize the control action Q_in(t), such that μ(t) can be estimated as precisely as possible. We restrict Q_in(t) to piecewise constant functions. This can be implemented in MTK as:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"@mtkmodel Bioreactor begin\n    @constants begin\n        C_s_in = 50.0\n        y_x_s = 0.777\n        m = 0.0\n    end\n    @parameters begin\n        controls[1:length(optimization_state)-1] = optimization_state[2:end], [tunable = false] # optimization_state is defined further below\n        Q_in = optimization_initial, [tunable = false] # similar for optimization state\n    end\n    @variables begin\n        C_s(t) = 1.0\n        C_x(t) = 1.0\n        V(t) = 7.0\n        μ(t)\n        σ(t)\n    end\n    @equations begin\n        σ ~ μ / y_x_s + m\n        D(C_s) ~ -σ * C_x + Q_in / V * (C_s_in - C_s)\n        D(C_x) ~ μ * C_x - Q_in / V * C_x\n        D(V) ~ Q_in\n    end\n    @discrete_events begin\n        (t == 1.0) => [Q_in ~ controls[1]]\n        (t == 2.0) => [Q_in ~ controls[2]]\n        (t == 3.0) => [Q_in ~ controls[3]]\n        (t == 4.0) => [Q_in ~ controls[4]]\n        (t == 5.0) => [Q_in ~ controls[5]]\n        (t == 6.0) => [Q_in ~ controls[6]]\n        (t == 7.0) => [Q_in ~ controls[7]]\n        (t == 8.0) => [Q_in ~ controls[8]]\n        (t == 9.0) => [Q_in ~ controls[9]]\n        (t == 10.0) => [Q_in ~ controls[10]]\n        (t == 11.0) => [Q_in ~ controls[11]]\n        (t == 12.0) => [Q_in ~ controls[12]]\n        (t == 13.0) => [Q_in ~ controls[13]]\n        (t == 14.0) => [Q_in ~ controls[14]]\n        (t == 15.0) => [Q_in ~ optimization_initial] # HACK TO GET Q_IN BACK TO ITS ORIGINAL VALUE\n    end\nend\nnothing # hide","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The true value of μ(t), which must be recovered is the Monod equation.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"beginequation*\nmu(C_s) = fracmu_maxC_sK_s + C_s\nendequation*","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"We thus extend the bioreactor MTK model with this equation:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"@mtkmodel TrueBioreactor begin\n    @extend Bioreactor()\n    @parameters begin\n        μ_max = 0.421\n        K_s = 0.439*10\n    end\n    @equations begin\n        μ ~ μ_max * C_s / (K_s + C_s) \n    end\nend\nnothing # hide","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"Similarly, we can extend the bioreactor with a neural network to represent this missing physics.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"@mtkmodel UDEBioreactor begin\n    @extend Bioreactor()\n    @structural_parameters begin\n        chain = Lux.Chain(Lux.Dense(1, 5, tanh),\n                          Lux.Dense(5, 5, tanh),\n                          Lux.Dense(5, 1, x->1*sigmoid(x)))\n    end\n    @components begin\n        nn = NeuralNetworkBlock(; n_input=1, n_output=1, chain, rng)\n    end\n    @equations begin\n        nn.output.u[1] ~ μ\n        nn.input.u[1] ~ C_s\n    end\nend\nnothing # hide","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"We start by gathering some initial data. Because we don't yet know anything about the missing physics, we arbitrarily pick the zero control action. The only state we measure is C_s We also add some noise to the simulated data, to make it more realistic:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"optimization_state =  zeros(15)\noptimization_initial = optimization_state[1] # HACK CAN'T GET THIS TO WORK WITHOUT SEPARATE SCALAR\n@mtkbuild true_bioreactor = TrueBioreactor()\nprob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)\nsol = solve(prob, Rodas5P())\n\n@mtkbuild  ude_bioreactor = UDEBioreactor()\nude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)\nude_sol = solve(ude_prob, Rodas5P())\n\ndata = DataFrame(sol)\ndata = data[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED; MTK ALWAYS SAVES BEFORE AND AFTER CALLBACK; WITH NO OPTION TO DISABLE\n\nsd_cs = 0.1\ndata[!, \"C_s(t)\"] += sd_cs * randn(size(data, 1))\n\nplts = plot(), plot(), plot(), plot()\nplot!(plts[1], sol, idxs=:C_s, lw=3,c=1)\nplot!(plts[1], ylabel=\"Cₛ(g/L)\", xlabel=\"t(h)\")\nscatter!(plts[1], data[!, \"timestamp\"], data[!, \"C_s(t)\"]; ms=3,c=1)\nplot!(plts[2], sol, idxs=:C_x, lw=3,c=1)\nplot!(plts[2], ylabel=\"Cₓ(g/L)\", xlabel=\"t(h)\")\nplot!(plts[3], sol, idxs=:V, ylabel=\"V(L)\", xlabel=\"t(h)\", lw=3, color=:black, ylims=(6.0,8.0))\nC_s_range_plot = 0.0:0.01:50.0\nμ_max = 0.421; K_s = 0.439*10 # TODO extract the  values from the model.\nplot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)\nplot!(plts[4], ylabel=\"μ(1/h)\", xlabel=\"Cₛ(g/L)\",ylims=(0,0.5))\nplot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"Now we can train the neural network to match this data:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"function loss(x, (probs, get_varss, datas))\n    loss = zero(eltype(x))\n    for i in eachindex(probs)\n        prob = probs[i]\n        get_vars = get_varss[i]\n        data = datas[i]\n        new_p = SciMLStructures.replace(Tunable(), prob.p, x)\n        new_prob = remake(prob, p=new_p, u0=eltype(x).(prob.u0))\n        new_sol = solve(new_prob, Rodas5P())\n        for (i, j) in enumerate(1:2:length(new_sol.t)) # HACK TO DEAL WITH DOUBLE SAVE\n            loss += sum(abs2.(get_vars(new_sol, j) .- data[!, \"C_s(t)\"][i]))\n        end\n        if !(SciMLBase.successful_retcode(new_sol))\n            println(\"failed\")\n            return Inf\n        end\n    end\n    loss\nend\nof = OptimizationFunction{true}(loss, AutoZygote())\nx0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))\nget_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])\nps = ([ude_prob], [get_vars], [data]);\nop = OptimizationProblem(of, x0, ps)\nres = solve(op, Optimization.LBFGS(), maxiters=1000)\n\nnew_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)\nres_prob = remake(ude_prob, p=new_p)\nres_sol = solve(res_prob, Rodas5P())\n\nextracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]\nT = defaults(ude_bioreactor)[ude_bioreactor.nn.T]\nμ_predicted_plot = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]\nμ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, \"C_s(t)\"]]\n\nplts = plot(), plot(), plot(), plot()\nplot!(plts[1], sol, idxs=:C_s, lw=3,c=1)\nplot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)\nplot!(plts[1], ylabel=\"Cₛ(g/L)\", xlabel=\"t(h)\")\nscatter!(plts[1], data[!, \"timestamp\"], data[!, \"C_s(t)\"]; ms=3,c=1)\nplot!(plts[2], sol, idxs=:C_x, lw=3,c=1)\nplot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)\nplot!(plts[2], ylabel=\"Cₓ(g/L)\", xlabel=\"t(h)\")\nplot!(plts[3], sol, idxs=:V, ylabel=\"V(L)\", xlabel=\"t(h)\", lw=3, color=:black, ylims=(6.0,8.0))\nplot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)\nplot!(plts[4], C_s_range_plot, μ_predicted_plot, lw=3, c=2)\nscatter!(plts[4], data[!, \"C_s(t)\"], μ_predicted_data, ms=3, c=2)\nplot!(plts[4], ylabel=\"μ(1/h)\", xlabel=\"Cₛ(g/L)\",ylims=(0,0.5))\nplot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"On the above figure we see that the neural network predicts C_s well, except during the final hours of the experiment, where we have multiple positive realizations of the noise in a row. The neural network also predicts µ well in the low substrate concentration region, where we have data available. However, the fit is poor at higher substrate concentrations, where we do not have data.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"We continue by making the neural network interpretable using symbolic regression.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"options = SymbolicRegression.Options(\n    unary_operators=(exp, sin, cos),\n    binary_operators=(+, *, /, -),\n    seed=123,\n    deterministic=true,\n    save_to_file=false,\n    defaults=v\"0.24.5\"\n)\nhall_of_fame = equation_search(collect(data[!, \"C_s(t)\"])', μ_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"Next, we extract the 10 model structures which symbolic regression thinks are best, and predict the system with them.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"n_best = 10\nfunction get_model_structures(hall_of_fame, options, n_best)\n    best_models = []\n    best_models_scores = []\n    i = 1\n    round(hall_of_fame.members[i].loss,sigdigits=5)\n    while length(best_models) <= n_best\n        member = hall_of_fame.members[i]\n        rounded_score = round(member.loss, sigdigits=5)\n        if !(rounded_score in best_models_scores)\n            push!(best_models,member)\n            push!(best_models_scores, rounded_score)\n        end\n        i += 1\n    end\n    model_structures = []\n    @syms x\n    for i = 1:n_best\n        eqn = node_to_symbolic(best_models[i].tree, options, varMap=[\"x\"])\n        fi = build_function(eqn, x, expression=Val{false})\n        push!(model_structures, fi)\n    end\n    return model_structures\nend\n\nfunction get_probs_and_caches(model_structures)\n    probs_plausible = Array{Any}(undef, length(model_structures))\n    syms_cache = Array{Any}(undef, length(model_structures))\n    i = 1\n    for i in 1:length(model_structures)\n        @mtkmodel PlausibleBioreactor begin\n            @extend Bioreactor()\n            @equations begin\n                μ ~ model_structures[i](C_s)\n            end\n        end\n        @mtkbuild plausible_bioreactor = PlausibleBioreactor()\n        plausible_prob = ODEProblem(plausible_bioreactor, [], (0.0, 15.0), [], tstops=0:15, saveat=0:15)\n        probs_plausible[i] = plausible_prob\n\n        callback_controls = plausible_bioreactor.controls\n        initial_control = plausible_bioreactor.Q_in\n\n        syms_cache[i] = (callback_controls, initial_control, plausible_bioreactor.C_s)\n    end\n    probs_plausible, syms_cache\nend\nmodel_structures = get_model_structures(hall_of_fame, options, n_best)\nprobs_plausible, syms_cache = get_probs_and_caches(model_structures)\n\nplts = plot(), plot(), plot(), plot()\nfor i in 1:length(model_structures)\n    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)\n    plausible_prob = probs_plausible[i]\n    sol_plausible = solve(plausible_prob, Rodas5P())\n    # plot!(sol_plausible; label=[\"Cₛ(g/L)\" \"Cₓ(g/L)\" \"V(L)\"], xlabel=\"t(h)\", lw=3)\n    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)\n    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)\nend\nplot!(plts[1], sol, idxs=:C_s, lw=3,c=1)\nplot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)\nplot!(plts[1], ylabel=\"Cₛ(g/L)\", xlabel=\"t(h)\")\nscatter!(plts[1], data[!, \"timestamp\"], data[!, \"C_s(t)\"]; ms=3,c=1)\nplot!(plts[2], sol, idxs=:C_x, lw=3,c=1)\nplot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)\nplot!(plts[2], ylabel=\"Cₓ(g/L)\", xlabel=\"t(h)\")\nplot!(plts[3], sol, idxs=:V, ylabel=\"V(L)\", xlabel=\"t(h)\", lw=3, color=:black, ylims=(6.0,8.0))\nμ_max = 0.421; K_s = 0.439*10 # TODO extract the  values from the model.\nplot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)\nplot!(plts[4], C_s_range_plot, μ_predicted_plot, lw=3, c=2)\nscatter!(plts[4], data[!, \"C_s(t)\"], μ_predicted_data, ms=3, c=2)\nplot!(plts[4], ylabel=\"μ(1/h)\", xlabel=\"Cₛ(g/L)\",ylims=(0,0.5))\nplot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"On the figure, we see that most plausible model structures predict the states C_s and C_x well, similar to the neural network. The plausible model structures also fit mu well in the low C_s region, but not outside this region. One group of the structures predicts that mu keeps increasing as C_s becomes large while another group predicts that mu stays below 01 1mathrmh.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"We now design a second experiment to start discriminating between these plausible model structures, using the following criterion:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"beginequation*\nargmax_bm Q_in frac2(10-2)10sum_i=1^10 sum_j=i+1^10 max_t_k (bm C_s^i(t_k) - bm C_s^j(t_k))^2\nendequation*","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"In this equation, C_s^i denotes the predicted substrate concentration for the i'th plausible model structure. The distance between two model structures is scored by the maximal squared difference between the two structures at the measurement times. The criterion then calculates the average distance between all model structures. Collecting measurements where the plausible model structures differ greatly in predictions, will cause at least some of the model structures to become unlikely, and thus cause new model structures to enter the top 10 plausible model structures.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"function S_criterion(optimization_state, (probs_plausible, syms_cache))\n    n_structures = length(probs_plausible)\n    sols = Array{Any}(undef, n_structures)\n    for i in 1:n_structures\n        plausible_prob = probs_plausible[i]\n        callback_controls, initial_control, C_s = syms_cache[i]\n        plausible_prob.ps[callback_controls] = optimization_state[2:end]\n        plausible_prob.ps[initial_control] = optimization_state[1]\n        sol_plausible = solve(plausible_prob, Rodas5P())\n        if !(SciMLBase.successful_retcode(sol_plausible))\n            return 0.0\n        end\n    loss\n        sols[i] = sol_plausible\n    end\n    squared_differences = Float64[]\n    for i in 1:n_structures\n        callback_controls, initial_control, C_s = syms_cache[i]\n        for j in i+1:n_structures\n            push!(squared_differences, maximum((sols[i][C_s] .- sols[j][C_s]) .^ 2))\n        end\n    end\n    ret = -mean(squared_differences)\n    println(ret)\n    return ret\nend\nlb = zeros(15)\nub = 10 * ones(15)\n\ndesign_prob = OptimizationProblem(S_criterion, optimization_state, (probs_plausible, syms_cache), lb=lb, ub=ub)\ncontrol_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=100.0)\n\noptimization_state = control_pars_opt.u\noptimization_initial = optimization_initial2 = optimization_state[1]\n\nplts = plot(), plot()\nt_pwc = []\npwc = []\nfor i in 0:14\n    push!(t_pwc,i)\n    push!(t_pwc,i+1)\n    push!(pwc,optimization_state[i+1])\n    push!(pwc,optimization_state[i+1])\nend\nplot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel=\"t(h)\",ylabel=\"Qin(L/h)\")\nfor i in 1:length(model_structures)\n    plausible_prob = probs_plausible[i]\n    callback_controls, initial_control, C_s = syms_cache[i]\n    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]\n    plausible_prob.ps[initial_control] = control_pars_opt[1]\n    sol_plausible = solve(plausible_prob, Rodas5P())\n    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)\nend\nplot!(plts[2],xlabel=\"t(h)\",ylabel=\"Cₛ(g/L)\")\nplot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The above figure shows that a maximal control action is generally preferred. This causes the two aforementioned groups in the model structures to be easily discriminated from one another.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"We now gather a second dataset and perform the same exercise.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"@mtkbuild true_bioreactor2 = TrueBioreactor()\nprob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)\nsol2 = solve(prob2, Rodas5P())\n@mtkbuild ude_bioreactor2 = UDEBioreactor()\nude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [ude_bioreactor2.Q_in => optimization_initial], tstops=0:15, save_everystep=false)\nude_sol2 = solve(ude_prob2, Rodas5P())\nplot(ude_sol2[3,:])\nude_prob_remake = remake(ude_prob, p=ude_prob2.p)\nsol_remake = solve(ude_prob_remake, Rodas5P())\nplot(sol_remake[3,:])\nx0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))\n\nget_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])\n\ndata2 = DataFrame(sol2)\ndata2 = data2[1:2:end, :]\ndata2[!, \"C_s(t)\"] += sd_cs * randn(size(data2, 1))\n\nps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2]);\nop = OptimizationProblem(of, x0, ps)\nres = solve(op, NLopt.LN_BOBYQA, maxiters=5_000)\n\nnew_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)\nres_prob = remake(ude_prob2, p=new_p)\ncallback_controls, initial_control, C_s = syms_cache[1]\nres_prob.ps[initial_control] = optimization_initial2\nres_sol = solve(res_prob, Rodas5P())\nextracted_chain = arguments(equations(ude_bioreactor2.nn)[1].rhs)[1]\nT = defaults(ude_bioreactor2)[ude_bioreactor2.nn.T]\nμ_predicted_plot2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]\n\nμ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, \"C_s(t)\"]]\nμ_predicted_data2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, \"C_s(t)\"]]\n\ntotal_data = hcat(collect(data[!, \"C_s(t)\"]'), collect(data2[!, \"C_s(t)\"]'))\ntotal_predicted_data =  vcat(μ_predicted_data, μ_predicted_data2)\nhall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)\nmodel_structures = get_model_structures(hall_of_fame, options, n_best)\nprobs_plausible, syms_cache = get_probs_and_caches(model_structures);\n\nplts = plot(), plot(), plot(), plot()\nfor i in 1:length(model_structures)\n    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)\n    plausible_prob = probs_plausible[i]\n    sol_plausible = solve(plausible_prob, Rodas5P())\n    # plot!(sol_plausible; label=[\"Cₛ(g/L)\" \"Cₓ(g/L)\" \"V(L)\"], xlabel=\"t(h)\", lw=3)\n    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)\n    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)\nend\nplot!(plts[1], sol2, idxs=:C_s, lw=3,c=1)\nplot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)\nplot!(plts[1], ylabel=\"Cₛ(g/L)\", xlabel=\"t(h)\")\nscatter!(plts[1], data2[!, \"timestamp\"], data2[!, \"C_s(t)\"]; ms=3,c=1)\nplot!(plts[2], sol2, idxs=:C_x, lw=3,c=1)\nplot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)\nplot!(plts[2], ylabel=\"Cₓ(g/L)\", xlabel=\"t(h)\")\nplot!(plts[3], sol2, idxs=:V, ylabel=\"V(L)\", xlabel=\"t(h)\", lw=3, color=:black)\nplot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)\nplot!(plts[4], C_s_range_plot, μ_predicted_plot2, lw=3, c=2)\nscatter!(plts[4], data[!, \"C_s(t)\"], μ_predicted_data, ms=3, c=2)\nscatter!(plts[4], data2[!, \"C_s(t)\"], μ_predicted_data2, ms=3, c=2)\nplot!(plts[4], ylabel=\"μ(1/h)\", xlabel=\"Cₛ(g/L)\",ylims=(0,0.5))\nplot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The above shows the data analysis corresponding to this second experiment. Both the UDE and most of the plausible model structures predict the states well,","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The UDE and the plausible model structures also approximate the missing physics mu well in the region where we have gathered data. This means in the regions of low substrate concentration, with data coming primarily from the first experiment, and high substrate concentration, coming from the second experiment. However, we do not have any measurements at substrate concentrations between these two groups. This causes there to be substantial disagreement between the plausible model structures in the medium substrate concentration range.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"We now optimize the controls for a third experiment:","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)\ncontrol_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)\n\noptimization_state = control_pars_opt.u\noptimization_initial = optimization_state[1]\n\nplts = plot(), plot()\nt_pwc = []\npwc = []\nfor i in 0:14\n    push!(t_pwc,i)\n    push!(t_pwc,i+1)\n    push!(pwc,optimization_state[i+1])\n    push!(pwc,optimization_state[i+1])\nend\nplot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel=\"t(h)\",ylabel=\"Qin(L/h)\")\nfor i in 1:length(model_structures)\n    plausible_prob = probs_plausible[i]\n    callback_controls, initial_control, C_s = syms_cache[i]\n    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]\n    plausible_prob.ps[initial_control] = control_pars_opt[1]\n    sol_plausible = solve(plausible_prob, Rodas5P())\n    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)\nend\nplot!(plts[2],xlabel=\"t(h)\",ylabel=\"Cₛ(g/L)\")\nplot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The optimal design algorithm is also aware of this uncertainty at the medium concentration range, and aims to remedy this in the next experiment, as can be seen on the above figure. Using the first control action, the bioreactor substrate concentration gets pumped from a low substrate concentration level to a medium level. At this level, there is substantial disagreement between the plausible model structures, leading to substantial disagreement in predicted substrate concentrations. To keep the reactor at the medium substrate concentration range, while the biomass concentration increases rapidly, an increasing amount of substrate has to be pumped into the reactor every hour. This explains the staircase with increasing step heights form of the control function. After the staircase reaches the maximal control value, a zero control is used. Some model structures decrease more rapidly in substrate concentration than others.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"@mtkbuild true_bioreactor3 = TrueBioreactor()\nprob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)\nsol3 = solve(prob3, Rodas5P())\n@mtkbuild ude_bioreactor3 = UDEBioreactor()\nude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)\n\nx0 = reduce(vcat, getindex.((default_values(ude_bioreactor3),), tunable_parameters(ude_bioreactor3)))\n\nget_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])\n\ndata3 = DataFrame(sol3)\ndata3 = data3[1:2:end, :]\ndata3[!, \"C_s(t)\"] += sd_cs * randn(size(data3, 1))\n\nps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3]);\nop = OptimizationProblem(of, x0, ps)\nres = solve(op, Optimization.LBFGS(), maxiters=10_000)\nextracted_chain = arguments(equations(ude_bioreactor3.nn)[1].rhs)[1]\nT = defaults(ude_bioreactor3)[ude_bioreactor3.nn.T]\n\nμ_predicted_data = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, \"C_s(t)\"]]\nμ_predicted_data2 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, \"C_s(t)\"]]\nμ_predicted_data3 = [only(stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, \"C_s(t)\"]]\n\ntotal_data = hcat(collect(data[!, \"C_s(t)\"]'), collect(data2[!, \"C_s(t)\"]'), collect(data3[!, \"C_s(t)\"]'))\ntotal_predicted_data =  vcat(μ_predicted_data, μ_predicted_data2, μ_predicted_data3)\nhall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)\nbar(i->hall_of_fame.members[i].loss, 1:10, ylabel=\"loss\", xlabel=\"hall of fame member\", xticks=1:10)\nplot!(tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"The Monod equation (0419  ((x1 + 4300)  x1)) is member 7 of the hall of fame. All hall of fame members before it have visually higher loss, while all the members after it are indiscernible from it. This indicates that it is a good candidate for the true model structure.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"Symbolic regression sometimes finds the true model structure in a somewhat unusual form, like with a double division. This is because symbolic regression considers multiplication and division to have the same complexity.","category":"page"},{"location":"Optimal Data Gathering for Missing Physics/","page":"Optimal Data Gathering for Missing Physics.","title":"Optimal Data Gathering for Missing Physics.","text":"In this tutorial, we have shown that experimental design can be used to explore the state space of a dynamic system in a thoughtful way, such that missing physics can be recovered in an efficient manner.","category":"page"}]
}
