using JuMP
using GLPK
import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Statistics")
# Portfolio Optimization using Mean Absolute Deviation (MAD) in Julia
using CSV
using DataFrames
using Statistics

function mad_portfolio(assets::DataFrame; R=0.001, w_lb=0.0, w_ub=0.2)
    # Step 1: Calculate daily returns and mean return
    prices = Matrix(assets)
    returns = diff(log.(prices), dims=1)
    mean_return = vec(mean(returns, dims=1))
    T, n = size(returns)

    # Step 2: Create JuMP model
    model = Model(GLPK.Optimizer)

    @variable(model, w[1:n] >= w_lb)     # portfolio weights
    @variable(model, z[1:T] >= 0)        # MAD deviation variables

    @constraint(model, sum(w) == 1)      # weights must sum to 1
    @constraint(model, sum(w[j] * mean_return[j] for j = 1:n) >= R)  # return constraint

    # Diversification constraint: weights ≤ w_ub
    for j in 1:n
        @constraint(model, w[j] <= w_ub)
    end

    # MAD constraints
    for t in 1:T
        deviation = sum(w[j] * (returns[t, j] - mean_return[j]) for j = 1:n)
        @constraint(model, z[t] >=  deviation)
        @constraint(model, z[t] >= -deviation)
    end

    # Objective: Minimize average deviation
    @objective(model, Min, sum(z) / T)

    # Step 3: Solve
    optimize!(model)

    # Step 4: Output results
    weights = value.(w)
    return weights
end