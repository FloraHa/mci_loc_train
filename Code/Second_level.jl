"""
This file stores the second level
"""

function build_third_level_model(RT, Fortified_Arc, Arc_empty, PStart, CapHosp, Dist, Dis, IwH, Hosp, I, TMax, CapVeh, s)
    model_third_level = Model(Gurobi.Optimizer)
    set_silent(model_third_level)
    set_optimizer_attribute(model_third_level, "MIPFocus", 1)
    set_optimizer_attribute(model_third_level, "Heuristics", 0.1)
    set_optimizer_attribute(model_third_level, "TimeLimit", 300)
    

    T = 1:TMax
    valid_arcs = get_valid_arcs(Dist)
    invalid_arcs = get_invalid_arcs(Dist)

    # Define variables
    @variable(model_third_level, pwt[IwH, T] >= 0)
    @variable(model_third_level, psm[IwH, I, T] >= 0)
    @variable(model_third_level, phf[Hosp, T] >= 0)
    @variable(model_third_level, pf[T] >= 0)
    @variable(model_third_level, vw[I, T] >= 0)
    @variable(model_third_level, vsm[I, I, T] >= 0, Int)

    # Precompute values
    valid_jit_IwH = Dict{Tuple{String,String,Int},Bool}()
    for j in IwH, i in IwH, t in T
        time_index = t - Dist[j][i]
        if time_index >= 1
            valid_jit_IwH[(j,i,t)] = true
        end
    end
    valid_js_per_i_t_IwH = Dict{Tuple{String,Int}, Vector{String}}()
    for i in IwH, t in T
        valid_js_per_i_t_IwH[(i,t)] = [j for j in IwH if t - Dist[j][i] >= 1]
    end
    valid_jit_I = Dict{Tuple{String,String,Int},Bool}()
    for j in I, i in I, t in T
        time_index = t - Dist[j][i]
        if time_index >= 1
            valid_jit_I[(j,i,t)] = true
        end
    end
    valid_js_per_i_t_I = Dict{Tuple{String, Int}, Vector{String}}()
    for i in I, t in T
        valid_js_per_i_t_I[(i,t)] = [j for j in I if t - Dist[j][i] >= 1]
    end

    total_RT = sum(RT[i] for i in I)

    # Objective
    @objective(model_third_level, Max, sum(pf[t] * (TMax-t) for t in T))

   
    P_init = Dict{String, Int}()
    for i in IwH
        P_init[i] = i in Dis ? PStart[i][s] : 0
    end

    @constraint(model_third_level, [i in IwH],
        pwt[i,1] == P_init[i] - sum(psm[i,j,1] for j in I)
    )

    @constraint(model_third_level, [i in IwH, t in 2:TMax],
        pwt[i,t] == pwt[i,t-1]
                    - sum(psm[i,j,t] for j in I)
                    + sum(psm[j,i,t - Dist[j][i]]
                        for j in valid_js_per_i_t_IwH[(i,t)])
    )

    # Update the number of casualties per time unit
    @constraint(model_third_level, [t in T],
        pf[t] == sum(psm[j, h, t-Dist[j][h]] for j in IwH, h in Hosp if t - Dist[j][h] >= 1)
    )

    # Update the number of casualties in each hospital per time unit 
    @constraint(model_third_level, [h in Hosp], phf[h,1] <= 0)
    @constraint(model_third_level, [h in Hosp, t in 2:TMax],
        phf[h,t] == phf[h,t-1] + sum(psm[j, h, t-Dist[j][h]] for j in IwH if t - Dist[j][h] >= 1)
    )
    @constraint(model_third_level, [j in Hosp, t in T], phf[j,t] <= CapHosp[j][s])

    # Vehicles constraints
    vw_init_constraints = @constraint(model_third_level, [i in I],
        vw[i,1] == RT[i] - sum(vsm[i,j,1] for j in I)
    )
    @constraint(model_third_level, [i in I, t in 2:TMax],
        vw[i,t] == vw[i,t-1] - sum(vsm[i,j,t] for j in I) + sum(vsm[j,i,t-Dist[j][i]] for j in valid_js_per_i_t_I[(i,t)])
    )

    # Arc-dependent constraints (placeholder for Arc and Fortified_Arc)
    arc_constraints = Dict{Tuple{String,String,Int}, ConstraintRef}()
    for (i, j) in valid_arcs, t in T
        arc_constraints[(i,j,t)] = @constraint(model_third_level,
            vsm[i,j,t] <= (1 - Arc_empty[i][j][s] + Fortified_Arc[i][j]) * total_RT  # Placeholder: Arc[i][j] will be updated later
        )
    end

    @constraint(model_third_level, [(i, j) in invalid_arcs, t in T], vsm[i,j,t] == 0)

    # Vehicle-patient interaction
    @constraint(model_third_level, [i in IwH, j in I, t in T],
        psm[i,j,t] <= vsm[i,j,t] * CapVeh
    )

    return (
        model = model_third_level,
        vw_init_constraints = vw_init_constraints,
        arc_constraints = arc_constraints,
        total_RT = total_RT,
        valid_arcs = valid_arcs,
        invalid_arcs = invalid_arcs,
        T = T
    )
end

function update_RT_constraints(model_data, RT)
    vw_init_constraints = model_data.vw_init_constraints
    for i in keys(RT)
        set_normalized_rhs(vw_init_constraints[i], RT[i])
    end
end

function update_third_level_model_for_iteration(model_data, Arc, Fortified_Arc)
    model_third_level = model_data.model
    arc_constraints = model_data.arc_constraints
    total_RT = model_data.total_RT
    valid_arcs = model_data.valid_arcs
    T = model_data.T

    # Update Arc and Fortified_Arc for the current iteration
    for (i, j) in valid_arcs, t in T
        rhs = (1 - Arc[i][j] + Fortified_Arc[i][j]) * total_RT
        set_normalized_rhs(arc_constraints[(i,j,t)], rhs)
    end
end

function solve_third_level_model(model_data)
    model_third_level = model_data.model

    optimize!(model_third_level)

    if termination_status(model_third_level) == MOI.INFEASIBLE
        return (status = :infeasible_unbounded, obj = nothing, vsm_val = nothing)
    end

    primal_obj = objective_value(model_third_level)
    vsm_val = value.(model_data.model[:vsm])

    return (
        status = :feasible,
        obj = primal_obj,
        vsm_val = vsm_val
    )
end

###################################################################################################
# Main column and constraint generation function

function level_2_ccg_scenarios(
    RT, Fortified_arc, Arc_empty, BudgetAttack, I, Dis, IwH, Hosp, Dist, PStart, CapHosp, TMax, s, model_data_per_scenario, g, ntoi, iton, shortest_paths, node_connections, beta)

    min_train_time_baseline, min_hosp_time_baseline = sum_min_travel_times(g, ntoi, RT, Dis, Hosp, PStart, CapHosp, s, ())
    baseline_total = min_train_time_baseline + min_hosp_time_baseline
    shortest_path_cache = Dict{Tuple{String,String}, Tuple{Float64, Float64, Float64}}()
    empty!(shortest_path_cache)

    # Initialize bounds
    LB = -Inf
    UB = Inf
    k = 1
    optimal = false

    # Master problem setup
    model_second_level = Model(Gurobi.Optimizer)
    set_silent(model_second_level)

    # Variables for master problem
    @variable(model_second_level, 0 <= x[I,I] <= 1, Bin)
    @variable(model_second_level, theta >= 0)

    # Objective: To be updated
    @objective(model_second_level, Min, theta)

    # Constraint: Respect the attack budget
    @constraint(model_second_level, sum(x[i,j] for i in I, j in I) <= BudgetAttack * 2)
    @constraint(model_second_level, [i in I, j in I], x[i,j] == x[j,i])
    @constraint(model_second_level, [i in I, j in I], x[i,j] <= 1 - Fortified_arc[i][j])
    @constraint(model_second_level, [i in Hosp, j in I], x[i,j] == 0)
    @constraint(model_second_level, [i in I, j in Hosp], x[i,j] == 0)

    Arc = deepcopy(Arc_empty)
    Best_attack_plan = deepcopy(Arc_empty)

    update_RT_constraints(model_data_per_scenario[s], RT)
    
    # Iterative loop for CCG
    while !optimal
        # Solve the master problem
        optimize!(model_second_level)
        if termination_status(model_second_level) != MOI.OPTIMAL
            return (obj = UB,
                    attacked_arcs = Best_attack_plan)
        end

        # Update lower bound
        LB = objective_value(model_second_level)

        attack_plan = OrderedDict{String, OrderedDict{String, Int64}}()

        for i in I
            attack_plan[i] = OrderedDict{String, Int64}()
            for j in I
                attack_plan[i][j] = Int(round(value(x[i, j])))
            end
        end

        # Apply the attack to the network
        Arc = deepcopy(attack_plan)

        # Solve the third-level problem for the given attack plan
        update_third_level_model_for_iteration(model_data_per_scenario[s], Arc, Fortified_arc)
        third_level_result = solve_third_level_model(model_data_per_scenario[s])

        if third_level_result.status == :infeasible_unbounded
            disconnecting_set = [(i,j) for i in I, j in I if attack_plan[i][j] == 1]
            @constraint(model_second_level,
                sum(x[i,j] for (i,j) in disconnecting_set) <= length(disconnecting_set) - 1)
            continue
        end

        third_level_obj = third_level_result.obj  # Survival probability after the attack

        # Update the upper bound
        if third_level_obj < UB
            Best_attack_plan = deepcopy(attack_plan)
        end
        UB = min(UB, third_level_obj)
    
        local freq = get_most_used_arcs_with_t(third_level_result.vsm_val, I, TMax)
        local used_arcs = filter_used_arcs(freq)
        local normalised_arcs = normalise(used_arcs)

        candidate_arcs = Dict()
        for (i, j) in keys(normalised_arcs)
            if i in IwH && j in IwH && Fortified_arc[i][j] == 0
                candidate_arcs[(i, j)] = normalised_arcs[(i, j)]
            end
        end

        @constraint(model_second_level, sum(normalised_arcs[(i,j)] * x[i,j] for (i,j) in keys(candidate_arcs) if attack_plan[i][j] == 0) >= 1)

        @constraint(model_second_level, theta >= third_level_obj + sum(-beta * (1/node_connections[i] * 1/node_connections[j]) * x[i,j] for (i,j) in keys(candidate_arcs) if attack_plan[i][j] == 0))


        if termination_status(model_second_level) == MOI.OPTIMAL && sp_obj <= objective_value(model_second_level)
            optimal = true
        end

        k += 1
    end

    return (
        obj = UB,
        attacked_arcs = Best_attack_plan
        )
end