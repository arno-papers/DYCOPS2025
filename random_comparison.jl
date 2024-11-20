using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("definitions.jl")
function loss3(x, (prob1, prob2, prob3, get_vars1, get_vars2, get_vars3, data1, data2, data3))
    new_p1 = SciMLStructures.replace(Tunable(), prob1.p, x)
    new_prob1 = remake(prob1, p=new_p1, u0=eltype(x).(prob1.u0))

    new_p2 = SciMLStructures.replace(Tunable(), prob2.p, x)
    new_prob2 = remake(prob2, p=new_p2, u0=eltype(x).(prob2.u0))

    new_p3 = SciMLStructures.replace(Tunable(), prob3.p, x)
    new_prob3 = remake(prob3, p=new_p3, u0=eltype(x).(prob3.u0))

    new_sol1 = solve(new_prob1, Rodas5P())
    new_sol2 = solve(new_prob2, Rodas5P())
    new_sol3 = solve(new_prob3, Rodas5P())
    if !(SciMLBase.successful_retcode(new_sol1) & SciMLBase.successful_retcode(new_sol2) & SciMLBase.successful_retcode(new_sol3))
        println("failed")
        return Inf
    end
    loss = zero(eltype(x))
    for (i, j) in enumerate(1:2:length(new_sol1.t))
        # @info "i: $i j: $j"
        loss += sum(abs2.(get_vars1(new_sol1, j) .- data1[!, "C_s(t)"][i]))
        loss += sum(abs2.(get_vars2(new_sol2, j) .- data2[!, "C_s(t)"][i]))
        loss += sum(abs2.(get_vars2(new_sol3, j) .- data3[!, "C_s(t)"][i]))
    end
    println(loss)
    loss
end

# Random experiment 1
optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor1 = TrueBioreactor()
prob1 = ODEProblem(true_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor1 = UDEBioreactor()
ude_prob1 = ODEProblem(ude_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)


optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor3 = TrueBioreactor()
prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

sol1 = solve(prob1, Rodas5P())
data1 = DataFrame(sol1)
data1 = data1[1:2:end, :]
data1[!, "C_s(t)"] += sd_cs * randn(size(data1, 1))
sol2 = solve(prob2, Rodas5P())
data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))
sol3 = solve(prob3, Rodas5P())
data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))
ude_sol1 = solve(ude_prob1, Rodas5P())
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_sol3 = solve(ude_prob3, Rodas5P())
get_vars1 = getu(ude_bioreactor1, [ude_bioreactor1.C_s])
get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])
get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

of = OptimizationFunction{true}(loss3, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor1),), tunable_parameters(ude_bioreactor1)))
ps = (ude_prob1, ude_prob2, ude_prob3, get_vars1, get_vars2, get_vars3, data1, data2, data3);

op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

extracted_chain = arguments(equations(ude_bioreactor1.nn)[1].rhs)[1]
T = defaults(ude_bioreactor1)[ude_bioreactor1.nn.T]
μ_predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data1[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data1, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

# Random experiment 2
optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor1 = TrueBioreactor()
prob1 = ODEProblem(true_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor1 = UDEBioreactor()
ude_prob1 = ODEProblem(ude_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)


optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor3 = TrueBioreactor()
prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

sol1 = solve(prob1, Rodas5P())
data1 = DataFrame(sol1)
data1 = data1[1:2:end, :]
data1[!, "C_s(t)"] += sd_cs * randn(size(data1, 1))
sol2 = solve(prob2, Rodas5P())
data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))
sol3 = solve(prob3, Rodas5P())
data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))
ude_sol1 = solve(ude_prob1, Rodas5P())
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_sol3 = solve(ude_prob3, Rodas5P())
get_vars1 = getu(ude_bioreactor1, [ude_bioreactor1.C_s])
get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])
get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

of = OptimizationFunction{true}(loss3, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor1),), tunable_parameters(ude_bioreactor1)))
ps = (ude_prob1, ude_prob2, ude_prob3, get_vars1, get_vars2, get_vars3, data1, data2, data3);

op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

extracted_chain = arguments(equations(ude_bioreactor1.nn)[1].rhs)[1]
T = defaults(ude_bioreactor1)[ude_bioreactor1.nn.T]
μ_predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data1[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data1, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

# Random experiment 3
optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor1 = TrueBioreactor()
prob1 = ODEProblem(true_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor1 = UDEBioreactor()
ude_prob1 = ODEProblem(ude_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)


optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor3 = TrueBioreactor()
prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

sol1 = solve(prob1, Rodas5P())
data1 = DataFrame(sol1)
data1 = data1[1:2:end, :]
data1[!, "C_s(t)"] += sd_cs * randn(size(data1, 1))
sol2 = solve(prob2, Rodas5P())
data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))
sol3 = solve(prob3, Rodas5P())
data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))
ude_sol1 = solve(ude_prob1, Rodas5P())
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_sol3 = solve(ude_prob3, Rodas5P())
get_vars1 = getu(ude_bioreactor1, [ude_bioreactor1.C_s])
get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])
get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

of = OptimizationFunction{true}(loss3, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor1),), tunable_parameters(ude_bioreactor1)))
ps = (ude_prob1, ude_prob2, ude_prob3, get_vars1, get_vars2, get_vars3, data1, data2, data3);

op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

extracted_chain = arguments(equations(ude_bioreactor1.nn)[1].rhs)[1]
T = defaults(ude_bioreactor1)[ude_bioreactor1.nn.T]
μ_predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data1[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data1, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

# Random experiment 4
optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor1 = TrueBioreactor()
prob1 = ODEProblem(true_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor1 = UDEBioreactor()
ude_prob1 = ODEProblem(ude_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)


optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor3 = TrueBioreactor()
prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

sol1 = solve(prob1, Rodas5P())
data1 = DataFrame(sol1)
data1 = data1[1:2:end, :]
data1[!, "C_s(t)"] += sd_cs * randn(size(data1, 1))
sol2 = solve(prob2, Rodas5P())
data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))
sol3 = solve(prob3, Rodas5P())
data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))
ude_sol1 = solve(ude_prob1, Rodas5P())
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_sol3 = solve(ude_prob3, Rodas5P())
get_vars1 = getu(ude_bioreactor1, [ude_bioreactor1.C_s])
get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])
get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

of = OptimizationFunction{true}(loss3, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor1),), tunable_parameters(ude_bioreactor1)))
ps = (ude_prob1, ude_prob2, ude_prob3, get_vars1, get_vars2, get_vars3, data1, data2, data3);

op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

extracted_chain = arguments(equations(ude_bioreactor1.nn)[1].rhs)[1]
T = defaults(ude_bioreactor1)[ude_bioreactor1.nn.T]
μ_predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data1[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data1, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

# Random experiment 5
optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor1 = TrueBioreactor()
prob1 = ODEProblem(true_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor1 = UDEBioreactor()
ude_prob1 = ODEProblem(ude_bioreactor1, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)


optimization_state =  rand(15)*10
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor3 = TrueBioreactor()
prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
@mtkbuild  ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)

sol1 = solve(prob1, Rodas5P())
data1 = DataFrame(sol1)
data1 = data1[1:2:end, :]
data1[!, "C_s(t)"] += sd_cs * randn(size(data1, 1))
sol2 = solve(prob2, Rodas5P())
data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))
sol3 = solve(prob3, Rodas5P())
data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))
ude_sol1 = solve(ude_prob1, Rodas5P())
ude_sol2 = solve(ude_prob2, Rodas5P())
ude_sol3 = solve(ude_prob3, Rodas5P())
get_vars1 = getu(ude_bioreactor1, [ude_bioreactor1.C_s])
get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])
get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

of = OptimizationFunction{true}(loss3, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor1),), tunable_parameters(ude_bioreactor1)))
ps = (ude_prob1, ude_prob2, ude_prob3, get_vars1, get_vars2, get_vars3, data1, data2, data3);

op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

extracted_chain = arguments(equations(ude_bioreactor1.nn)[1].rhs)[1]
T = defaults(ude_bioreactor1)[ude_bioreactor1.nn.T]
μ_predicted_data1 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data1[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data1[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data1, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)
