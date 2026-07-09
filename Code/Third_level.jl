function build_models_PH(RT, 
    Fortified_Arc, BudgetArcs, BudgetLocation, LocCap,
    Arc, Dist, Dis, IwH, Hosp, I, TMax, PStart, CapVeh, CapHosp, weight_dev_train, weight_dev_arc, s)

    model_third_level_PH = Model(Gurobi.Optimizer)
    set_silent(model_third_level_PH)
    set_optimizer_attribute(model_third_level_PH, "MIPFocus", 1)
    set_optimizer_attribute(model_third_level_PH, "Heuristics", 0.1)
    set_optimizer_attribute(model_third_level_PH, "TimeLimit", 300)

    T = 1:TMax
    valid_arcs = get_valid_arcs(Dist)
    invalid_arcs = get_invalid_arcs(Dist)

    # Define variables
    @variable(model_third_level_PH, pwt[IwH, T] >= 0)
    @variable(model_third_level_PH, psm[IwH, I, T] >= 0)
    @variable(model_third_level_PH, phf[Hosp, T] >= 0)
    @variable(model_third_level_PH, pf[T] >= 0)
    @variable(model_third_level_PH, vw[I, T] >= 0)
    @variable(model_third_level_PH, vsm[I, I, T] >= 0, Int)

    @variable(model_third_level_PH, rt[I] >= 0, Int)
    @variable(model_third_level_PH, protected_arcs[I, I] >= 0, Int)

    @variable(model_third_level_PH, z_rt_pos[i in I] >= 0)  # Auxiliary variables for RT deviation, 1 if train located to spot not used before
    @variable(model_third_level_PH, z_fortified_pos[i in I, j in I] >= 0)  # Auxiliary variables for Fortified Arc deviation, 1 if arc fortified not used before
    @variable(model_third_level_PH, z_rt_neg[i in I] >= 0)  # Auxiliary variables for RT deviation, 1 if train not located at spot used before
    @variable(model_third_level_PH, z_fortified_neg[i in I, j in I] >= 0)  # Auxiliary variables for Fortified Arc deviation, 1 if arc fortified not used before

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

    total_PStart_s = sum(PStart[l][s] for l in Dis)

    # Linearize the absolute value terms
    z_rt_constraints_1 = @constraint(model_third_level_PH, [i in I], z_rt_pos[i] - rt[i] >= -RT[i])
    z_rt_constraints_2 = @constraint(model_third_level_PH, [i in I], z_rt_neg[i] + rt[i] >=  RT[i])
    z_fortified_constraints_1 = @constraint(model_third_level_PH, [i in I, j in I], z_fortified_pos[i,j] - protected_arcs[i,j] >= - Fortified_Arc[i][j])
    z_fortified_constraints_2 = @constraint(model_third_level_PH, [i in I, j in I], z_fortified_neg[i,j] + protected_arcs[i,j] >=  Fortified_Arc[i][j])

    #############################################
    # Objective

    # @objective(model_third_level_PH, Max, sum(pf[t] * (TMax - t) for t in T) - sum(z_rt[i] for i in I) * weight_dev_train - sum(z_fortified[i,j] for i in I, j in I) * weight_dev_arc)
    @objective(model_third_level_PH, Max, sum(pf[t] * (TMax - t) for t in T) 
                - sum(z_rt_pos[i] + z_rt_neg[i] for i in I) * weight_dev_train 
                - sum(z_fortified_pos[i,j] + z_fortified_neg[i,j] for i in I, j in I) * weight_dev_arc)

    #############################################
    # New constraints

    @constraint(model_third_level_PH, sum(rt[i] for i in I) <= BudgetLocation)
    @constraint(model_third_level_PH, [i in I], rt[i] <= LocCap[i])

    # Respect budget for fortification
    @constraint(model_third_level_PH, sum(protected_arcs[i,j] for i in I, j in I) <= BudgetArcs * 2)
    @constraint(model_third_level_PH, [i in I, j in I], protected_arcs[i,j] == protected_arcs[j,i])

    #############################################
    # Casulaties constraints

    P_init = Dict{String, Int}()
    for i in IwH
        P_init[i] = i in Dis ? PStart[i][s] : 0
    end

    @constraint(model_third_level_PH, [i in IwH],
        pwt[i,1] == P_init[i] - sum(psm[i,j,1] for j in I)
    )

    @constraint(model_third_level_PH, [i in IwH, t in 2:TMax],
        pwt[i,t] == pwt[i,t-1]
                    - sum(psm[i,j,t] for j in I)
                    + sum(psm[j,i,t - Dist[j][i]]
                        for j in valid_js_per_i_t_IwH[(i,t)])
    )

    # Update the number of casualties per time unit
    @constraint(model_third_level_PH, [t in T],
        pf[t] == sum(psm[j, h, t-Dist[j][h]] for j in IwH, h in Hosp if t - Dist[j][h] >= 1)
    )

    # Update the number of casualties in each hospital per time unit 
    @constraint(model_third_level_PH, [h in Hosp], phf[h,1] <= 0)
    @constraint(model_third_level_PH, [h in Hosp, t in 2:TMax],
        phf[h,t] >= phf[h,t-1] + sum(psm[j, h, t-Dist[j][h]] for j in IwH if t - Dist[j][h] >= 1)
    )
    @constraint(model_third_level_PH, [j in Hosp, t in T], phf[j,t] <= CapHosp[j][s])

    #############################################
    # Vehicle constraints
    @constraint(model_third_level_PH, [i in I],
        vw[i,1] == rt[i] - sum(vsm[i,j,1] for j in I)
    )
    @constraint(model_third_level_PH, [i in I, t in 2:TMax],
        vw[i,t] == vw[i,t-1] - sum(vsm[i,j,t] for j in I) + sum(vsm[j,i,t-Dist[j][i]] for j in valid_js_per_i_t_I[(i,t)])
    )

    # Arc-dependent constraints (placeholder for Arc and Fortified_Arc)
    arc_constraints = Dict{Tuple{String,String,Int}, ConstraintRef}()
    for (i, j) in valid_arcs, t in T
        arc_constraints[(i,j,t)] = @constraint(model_third_level_PH,
            vsm[i,j,t]/BudgetLocation - protected_arcs[i, j] <= 1 - Arc[i][j][s]
        )
    end

    @constraint(model_third_level_PH, [(i, j) in invalid_arcs, t in T], vsm[i,j,t] == 0)

    # Vehicle-patient interaction
    @constraint(model_third_level_PH, [i in IwH, j in I, t in T],
        psm[i,j,t] <= vsm[i,j,t] * CapVeh
    )

    @constraint(model_third_level_PH, [t in T], sum(rt[i] for i in I) - sum(vw[i, t] for i in I) <= total_PStart_s)

    @constraint(model_third_level_PH, [i in I, j in I, t in T], vsm[i,j,t] <= total_PStart_s - sum(pf[t_n] for t_n in 1:t))

    return (
        model = model_third_level_PH,
        z_rt_constraints_1 = z_rt_constraints_1,
        z_rt_constraints_2 = z_rt_constraints_2,
        z_fortified_constraints_1 = z_fortified_constraints_1,
        z_fortified_constraints_2 = z_fortified_constraints_2,
        arc_constraints = arc_constraints,
        valid_arcs = valid_arcs,
        invalid_arcs = invalid_arcs,
        T = T,
        pf = pf,
        z_rt_pos = z_rt_pos,
        z_rt_neg = z_rt_neg,
        z_fortified_pos = z_fortified_pos,
        z_fortified_neg = z_fortified_neg,
    )
end

function update_RT_constraints_PH(model_data, RT)
    z_rt_constraints_1 = model_data.z_rt_constraints_1
    z_rt_constraints_2 = model_data.z_rt_constraints_2

    for i in keys(RT)
        # Update z_rt[i] >= rt[i] - RT[i]
        #set_normalized_coefficient(z_rt_constraints_1[i], model_data.model[:rt][i], 1.0)
        set_normalized_rhs(z_rt_constraints_1[i], -RT[i])

        # Update z_rt[i] >= -(rt[i] - RT[i])
        #set_normalized_coefficient(z_rt_constraints_2[i], model_data.model[:rt][i], -1.0)
        set_normalized_rhs(z_rt_constraints_2[i], RT[i])
    end

end

function update_FortifiedArc_constraints_PH(model_data, Fortified_Arc, Attacked_arc, BudgetLocation)
    z_fortified_constraints_1 = model_data.z_fortified_constraints_1
    z_fortified_constraints_2 = model_data.z_fortified_constraints_2
    arc_constraints = model_data.arc_constraints
    valid_arcs = model_data.valid_arcs
    T = model_data.T

    for i in keys(Fortified_Arc), j in keys(Fortified_Arc[i])
        # Update z_fortified[i,j] >= protected_arcs[i,j] - Fortified_Arc[i][j]
        set_normalized_rhs(z_fortified_constraints_1[i,j], -Fortified_Arc[i][j])

        # Update z_fortified[i,j] >= -(protected_arcs[i,j] - Fortified_Arc[i][j])
        set_normalized_rhs(z_fortified_constraints_2[i,j], Fortified_Arc[i][j])

        # Update arc constraints
        for t in T
            if (i, j) in valid_arcs
                rhs = 1 - Attacked_arc[i][j]
                set_normalized_rhs(arc_constraints[(i,j,t)], rhs)
            end
        end
    end
end


function update_objective!(model_data, norm_weight_loc, norm_weight_arc, weight_dev_train, weight_dev_arc)
    model = model_data.model
    pf = model_data.pf
    z_rt_pos = model_data.z_rt_pos
    z_rt_neg = model_data.z_rt_neg
    z_f_pos = model_data.z_fortified_pos
    z_f_neg = model_data.z_fortified_neg
    T = model_data.T
    TMax = maximum(T)

    @objective(model, Max,
          sum(pf[t] * (TMax - t) for t in T)
        - weight_dev_train * sum(z_rt_pos[i] * (1 - norm_weight_loc[i]) + z_rt_neg[i] * norm_weight_loc[i] for i in keys(norm_weight_loc))
        - weight_dev_arc   * sum(z_f_pos[i,j] * (1 - norm_weight_arc[(i,j)]) + z_f_neg[i,j] * norm_weight_arc[(i,j)] for (i,j) in keys(norm_weight_arc))
    )

end


function solve_model_PH(model_data)

    # Solve the model
    optimize!(model_data.model)

    if termination_status(model_data.model) == MOI.INFEASIBLE
        return (status = :infeasible_unbounded, obj = nothing, vsm_val = nothing)
    end

    primal_obj = objective_value(model_data.model)
    vsm_val = value.(model_data.model[:vsm])

    # Extract rt values
    rt_vars = filter(v -> startswith(String(name(v)), "rt"), all_variables(model_data.model))
    rt_values = OrderedDict(
        replace(String(name(v)), "rt[" => "", "]" => "") => Int(round(value(v)))
        for v in rt_vars
    )

    # Extract protected_arcs values
    arc_vars = filter(v -> startswith(String(name(v)), "protected_arcs"), all_variables(model_data.model))
    arc_values = OrderedDict{String, OrderedDict{String, Int64}}()
    for v in arc_vars
        name_parts = replace(String(name(v)), "protected_arcs[" => "", "]" => "")
        origin, destination = split(name_parts, ",")
        origin = strip(origin, ' ')
        destination = strip(destination, ' ')

        if !haskey(arc_values, origin)
            arc_values[origin] = OrderedDict{String, Int64}()
        end
        arc_values[origin][destination] = Int(round(value(v)))
    end

    # Extract and process vsm variables to track departures
    vsm_vars = filter(v -> startswith(String(name(v)), "vsm"), all_variables(model_data.model))
    departure_counts = Dict{String, Int64}()
    for v in vsm_vars
        name_parts = replace(String(name(v)), "vsm[" => "", "]" => "")
        parts = split(name_parts, ",")
        origin = strip(parts[1], ' ')
        vsm_value = Int(round(value(v)))

        if vsm_value > 0
            if haskey(departure_counts, origin)
                departure_counts[origin] += vsm_value
            else
                departure_counts[origin] = vsm_value
            end
        end
    end

    return (
        status = :feasible,
        obj = primal_obj,
        rt_new_val = rt_values,
        fortified_new_val  = arc_values,
        departure_counts = departure_counts, 
        vsm_values = value.(model_data.model[:vsm])
    )
end
