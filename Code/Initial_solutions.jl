"""
This file contains the functions to create the initial solution
"""

using JuMP
using Gurobi
using OrderedCollections
using StatsBase
using Graphs, MetaGraphs, GraphPlot
using SimpleWeightedGraphs
using Clustering, LinearAlgebra
using Random

include("Add_functions.jl")

###################################################################################################
# Initial solution where the relaxed MP is solved
###################################################################################################

function relaxed_facility_location(BudgetLocation, LocCap,  BudgetArcs, Arc, Dist,
    Dis, IwH, Hosp, I, TMax, NumScen, ScenProb, PStart, CapVeh, CapHosp; force_integer=false)
        
    master_model = Model(Gurobi.Optimizer)
    set_silent(master_model)
    set_optimizer_attribute(master_model, "MIPFocus", 1)
    set_optimizer_attribute(master_model, "Heuristics", 0.1)
    set_optimizer_attribute(master_model, "TimeLimit", 300)

    T = 1:TMax
    S = 1:NumScen

    valid_arcs = get_valid_arcs(Dist)
    invalid_arcs = get_invalid_arcs(Dist)

    # Define variable upper stage
    @variable(master_model, rt[I]>=0, integer=true, base_name="rt")
    @variable(master_model, 0 <= protected_arcs[I, I] <= 1, base_name="protected_arcs")


    # Define variables lowest level
    @variable(master_model, pwt[IwH, T, S]>=0, base_name="pwt")                                   
    @variable(master_model, psm[IwH, I, T, S]>=0, base_name="psm")
    @variable(master_model, phf[Hosp, T, S]>=0, base_name="phf")
    @variable(master_model, pf[T, S]>=0, base_name="pf")

    @variable(master_model, vsm[I, I, T, S]>=0, base_name="vsm")
    @variable(master_model, vw[I, T, S]>=0, base_name="vw")

    # Define linking variable
    @variable(master_model, ζ, base_name="ζ")

    # Precompute certain values
    valid_jits_IwH = Dict{Tuple{String,String,Int,Int},Bool}()

    for j in IwH, i in IwH, t in T, s in S
        time_index = t - Dist[j][i]  # assuming Dist is Dict{String,Dict{String,Int}} or similar
        if time_index >= 1
            valid_jits_IwH[(j,i,t,s)] = true
        end
    end

    valid_js_per_i_t_s_IwH = Dict{Tuple{String,Int,Int}, Vector{String}}()

    for i in IwH, t in T, s in S
        valid_js_per_i_t_s_IwH[(i,t,s)] = [j for j in IwH if t - Dist[j][i] >= 1]
    end

    valid_jits_I = Dict{Tuple{String,String,Int,Int},Bool}()

    for j in I, i in I, t in T, s in S
        time_index = t - Dist[j][i]  # assuming Dist is Dict{String,Dict{String,Int}} or similar
        if time_index >= 1
            valid_jits_I[(j,i,t,s)] = true
        end
    end

    valid_js_per_i_t_s_I = Dict{Tuple{String,Int,Int}, Vector{String}}()

    for i in I, t in T, s in S
        valid_js_per_i_t_s_I[(i,t,s)] = [j for j in I if t - Dist[j][i] >= 1]
    end

    total_PStart_s = Dict(s => sum(PStart[l][s] for l in Dis) for s in S)

    #############################################
    # Objective
    
    @objective(master_model, Max, ζ)

    #############################################
    # First level constraints

    # Respect budget for location
    @constraint(master_model, sum(rt[i] for i in I) <= BudgetLocation)
    @constraint(master_model, [i in I], rt[i] <= LocCap[i])

    # Respect budget for fortification
    @constraint(master_model, sum(protected_arcs[i,j] for i in I, j in I) <= BudgetArcs*2)
    @constraint(master_model, [i in I, j in I], protected_arcs[i,j] == protected_arcs[j,i])

    #############################################
    # Linking constraint
    @constraint(master_model, ζ== sum(pf[t, s] * (TMax-t) * ScenProb[s] for s in S, t in T))

    #############################################
    # Casualty constraints
    P_init = Dict{String, Vector{Int}}()
    for i in IwH
        P_init[i] = [ i in Dis ? PStart[i][s] : 0 for s in 1:NumScen ]
    end

    @constraint(master_model, [i in IwH, s in S],
        pwt[i,1,s] == P_init[i][s] - sum(psm[i,j,1,s] for j in I)
    )

    @constraint(master_model, [i in IwH, t in 2:TMax, s in S],
        pwt[i,t,s] == pwt[i,t-1,s]
                    - sum(psm[i,j,t,s] for j in I)
                    + sum(psm[j,i,t - Dist[j][i], s]
                        for j in valid_js_per_i_t_s_IwH[(i,t,s)])
    )


    # Update the number of casualties per time unit
    @constraint(master_model, [t in T, s in S],pf[t, s] == sum(psm[j, h, t-Dist[j][h], s] for j in IwH, h in Hosp if t - Dist[j][h] >= 1))

    # Update the number of casualites in each hospital per time unit
    @constraint(master_model, [h in Hosp, s in S], phf[h,1,s] <= 0)
    @constraint(master_model, [h in Hosp, t in 2:TMax, s in S], phf[h,t,s] == phf[h,t-1,s] + sum(psm[j, h, t-Dist[j][h], s] for j in IwH if t - Dist[j][h] >= 1))

    # Respect capacity
    @constraint(master_model, [j in Hosp, t in T, s in S], phf[j,t,s] <= CapHosp[j][s])

    #############################################
    # Vehicles constraints

    # Update the number of waiting vehicles
    @constraint(master_model, [i in I, s in S], vw[i,1,s] == rt[i] - sum(vsm[i,j,1, s] for j in I))
    @constraint(master_model, [i in I, t in 2:TMax, s in S], vw[i,t, s] == vw[i,t-1, s] - sum(vsm[i,j,t, s] for j in I) + sum(vsm[j,i,t-Dist[j][i], s] for j in valid_js_per_i_t_s_I[(i,t,s)]))


    # No moving on destroyed arcs
    @constraint(master_model, [(i, j) in valid_arcs, t in T, s in S], vsm[i,j,t,s] <= (1-Arc[i][j][s] + protected_arcs[i,j]) * BudgetLocation)

    # for i in I, j in I, t in T, s in S
    @constraint(master_model, [(i, j) in valid_arcs, t in T, s in S], vsm[i,j,t,s] <= BudgetLocation)

    @constraint(master_model, [(i, j) in invalid_arcs, s in S, t in T], vsm[i,j,t,s] == 0)

    # Restrict the number of moving vehicles
    @constraint(master_model, [t in T, s in S], sum(vsm[i, j, t, s] for i in I, j in I) <= total_PStart_s[s])


    #############################################
    # Vehicle patient interaction

    # The number of patients cannot exceed the capacity of the total vehicles
    @constraint(master_model, [i in IwH, j in I, t in T, s in S], psm[i,j,t,s] <= vsm[i,j,t,s] * CapVeh)

    # Dont move if all patients were served
    @constraint(master_model, [(i, j) in valid_arcs, t in T, s in S], vsm[i,j,t,s] <= total_PStart_s[s] - sum(pf[t_n,s] for t_n in 1:t))
    #############################################

    optimize!(master_model)

    rt_vars = filter(v -> startswith(name(v), "rt"), all_variables(master_model))
    rt_values = OrderedDict(
        replace(name(v), "rt[" => "", "]" => "") => Int(value(v)) 
        for v in rt_vars
    )

    # Extract and process "protected_arcs" variables into a nested OrderedDict
    arc_vars = filter(v -> startswith(name(v), "protected_arcs"), all_variables(master_model))
    arc_values = OrderedDict{String, OrderedDict{String, Int64}}()

    for v in arc_vars
        # Extract origin and destination nodes from the variable name
        name_parts = replace(name(v), "protected_arcs[" => "", "]" => "")
        origin, destination = split(name_parts, ",")
        
        # Initialize nested OrderedDict if not already present
        if !haskey(arc_values, origin)
            arc_values[origin] = OrderedDict{String, Int64}()
        end
        
        # Assign the variable value to the appropriate entry
        arc_values[origin][destination] = Int(value(v))
    end

    # Access the variable
    vsm_values = Dict{Tuple{String, String}, Float64}()

    # Aggregate over T and S
    for i in I, j in I
        total = 0.0
        for t in 1:TMax, s in 1:NumScen
            total += value(master_model[:vsm][i, j, t, s])
        end
        vsm_values[(i, j)] = total
    end

    return(
        obj = objective_value(master_model),
        rt_values = rt_values,
        arc_values = arc_values,
        vsm_values = vsm_values
    )
end

function RandomNodeSelection(BudgetLocation, LocCap)
    valid_locations = [k for (k, v) in LocCap if v > 0]

    sampled_locations = sample(valid_locations, min(BudgetLocation, length(valid_locations)), replace=false)

    sampled_locations_dict = Dict(k => (k in sampled_locations ? 1 : 0) for k in keys(LocCap))

    return sampled_locations_dict
end

###################################################################################################
# Arc protection
###################################################################################################

function EmptyFortification(I)
    local selected_arcs = Dict{Tuple{String, String}, Float64}()
    best_fortification = create_arc_values(selected_arcs, I)

    return best_fortification
end

function greedy_arc_protection(result, I, BudgetFortification)
    # This function protects the most used arcs
    local freq = get_most_used_arcs(result.vsm_values, I)
    local used_arcs = filter_used_arcs(freq)
    local unique_arcs = merge_arc_counts(used_arcs)
    local selected_arcs = select_top_arcs(unique_arcs, BudgetFortification)
    best_fortification = create_arc_values(selected_arcs, I)
    println("Selected arcs")
    println(selected_arcs)

    return best_fortification
end

###################################################################################################
# MCLP for the benefit
###################################################################################################

function sum_scenarios(I, PStart, ScenProb)
    """
    This function calculates the weighted demand per location
    """
    Weighted_demand = Dict()
    for i in I 
        demand = 0
        for scen in eachindex(ScenProb)
            prob = ScenProb[scen]
            dem_scen = PStart[i][scen]
            demand += prob * dem_scen
        end
        Weighted_demand[i] = demand
    end
    return Weighted_demand
end

function filter_demand_locations(Weighted_demand)
    """
    This function filters for the demand locations with a demand > 0
    """
    filtered_locations = filter(((k, v),) -> v > 0, Weighted_demand)
    return filtered_locations
end

function precalculate_coverage(shortest_paths, facilities, demand_loc, radius)
    Coverage = Dict()
    for demand in demand_loc
        Coverage[demand] = []
        for fac in facilities
            if shortest_paths[fac][demand] <= radius
                push!(Coverage[demand], fac)
            end
        end
    end
    return Coverage
end

function graph_shortest_path(Dis, Dist, CapLoc)
    # Step 1: Convert distances to a Graph representation
    g = Graphs.Graph(length(Dist))  # Create an undirected graph
    node_index = Dict(node => i for (i, node) in enumerate(keys(Dist)))
    
    # Add edges to the graph
    for (source, targets) in Dist
        for (target, cost) in targets
            if cost < 1000  # Only add valid edges
                Graphs.add_edge!(g, node_index[source], node_index[target])
            end
        end
    end

    # Step 2: Compute shortest paths from all potential locations to all demand sites
    placement_nodes = [k for (k, v) in CapLoc if v > 0]  # Sites that can receive vehicles
    demand_indices = [node_index[d] for d in Dis]
    shortest_paths = Dict()

    for p in placement_nodes
        source_idx = node_index[p]
        # Use Dijkstra to find the shortest paths from each placement to demand sites
        dijkstra_result = Graphs.dijkstra_shortest_paths(g, source_idx)
        dists = dijkstra_result.dists
        shortest_paths[p] = Dict(Dis[i] => dists[demand_indices[i]] for i in eachindex(Dis))
    end
    return shortest_paths
end

function MCLP(facilities, demand, coverage, budget)
    model  = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, covered[demand], binary=true, base_name="covered")
    @variable(model, site[facilities], binary=true, base_name="site")

    # Objective
    @objective(model, Max, sum(covered[dem] for dem in demand))

    # Constraints
    @constraint(model, sum(site[fac] for fac in facilities) == budget)

    for dem in demand
        @constraint(model, sum(site[fac] for fac in coverage[dem]) >= covered[dem])
    end

    optimize!(model)

    covered_vars = filter(v -> startswith(name(v), "covered"), all_variables(model))
    covered_values = OrderedDict(
        replace(name(v), "covered[" => "", "]" => "") => Int(value(v)) 
        for v in covered_vars
    )
    site_vars = filter(v -> startswith(name(v), "site"), all_variables(model))
    site_values = OrderedDict(
        replace(name(v), "site[" => "", "]" => "") => Int(value(v)) 
        for v in site_vars
    )

    return(
        obj = objective_value(model),
        covered_values, 
        site_values
    )

end

function convert_solution(result, I)
    locations = Dict()
    for loc in I
        if haskey(result.site_values, loc)
            locations[loc] = result.site_values[loc]
        else
            locations[loc] = 0
        end
    end
    return locations
end

function MCLP_sol(BudgetLoc, Radius, CapLoc, PStart, ScenProb, I, Dis, Dist)
    """
    In this function we first solve an MCLP to get an approximate solution
    """
    # Step 1: Calculate the weighted demand
    Weighted_demand = sum_scenarios(I, PStart, ScenProb)

    # Step 2: Filter for demand locations with a demand > 0
    filtered_weighted_demand = filter_demand_locations(Weighted_demand)

    # Step 3: Extract the demand locations for the optimization
    demand_loc = collect(keys(filtered_weighted_demand))

    # Step 4: Extract the locations for potential placement
    filtered_facility_locations = collect(keys(filter(((k, v), ) -> v > 0, CapLoc)))

    # Step 5: Create the graph to calculate the distances 
    shortest_paths = graph_shortest_path(Dis, Dist, CapLoc)

    # Step 6: Calculate the sets of coverage
    coverage = precalculate_coverage(shortest_paths, filtered_facility_locations, demand_loc, Radius)

    # Step 7: Determine the MCLP
    result = MCLP(filtered_facility_locations, demand_loc, coverage, BudgetLoc)

    # Step 8: Convert the solution
    locations = convert_solution(result, I)

    return locations
end

#####################################################################################################################################################
