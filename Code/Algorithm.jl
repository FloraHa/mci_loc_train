"""
This file contains the main algorithm implementation
"""

using JuMP
using Gurobi

include("Initial_solutions.jl")
include("Second_level.jl")
include("Third_level.jl")
include("Add_functions.jl")
include("Voting.jl")
include("Check_ending_criterion.jl")
include("Graph_generation.jl")


function get_edge_value(I, Dist)
    node_connections = Dict{String, Int}()
    for i in I
        for j in I
            if i != j
                if Dist[i][j] < 1000
                    node_connections[i] = get(node_connections, i, 0) + 1
                end
            end
        end
    end
    return node_connections
end

###################################################################################################
# Main algorithm
###################################################################################################

function main_algorithm(BudgetLocation, BudgetFortification, BudgetAttack,
        ArcsMP, data, Params_algorithm, TMax, NumScen, Cap_Veh)
    time = @time begin

        # Initialize the storage
        results_storage = OrderedDict()         # Contains the overall results
        results_sp_dict = OrderedDict()         # Contains the results from the second stage
        results_th_stage_dict = OrderedDict()   # Contains the results from the third stage (to reconsider the first stage solutions)

        counter = 1                             # Counts the current number of iterations

        # Create an empty dict of arcs
        Arc_empty = flatten_inner_dict_scenarios(ArcsMP) # Empty dict of arcs for initial solution

        # Create an empty dict of avg_weighted_arc_fortification
        empty_weighted_sums = create_empty_dict_weighted_sums(data)

        global relaxed_solution = Inf

        t1 = @elapsed begin

            # Determine the location and fortification
            if Params_algorithm["Initialisation"]["Method"] == "Simple"
                results_location = relaxed_facility_location(BudgetLocation, data[:CapLoc], BudgetFortification, ArcsMP, data[:Dist],
                data[:Dis], data[:IwH], data[:Hosp], data[:I], TMax, NumScen, data[:ScenProb], data[:PStart], Cap_Veh, data[:CapHosp], force_integer=true)
                best_location = deepcopy(results_location.rt_values)
                relaxed_solution = results_location.obj
                best_fortification = greedy_arc_protection(results_location, data[:I], 0)
            elseif Params_algorithm["Initialisation"]["Method"] == "Random"
                best_location = RandomNodeSelection(BudgetLocation, data[:CapLoc])
                best_fortification = EmptyFortification(data[:I])
            end
        end

        # Initialize the current solution
        current_location = deepcopy(best_location)
        current_fortification = deepcopy(best_fortification)
        best_obj = 0

        ###########################################################################################
        # Start the progressive hedging
        ###########################################################################################

        global stopping = false

        # Initialize the times for phase 2 and phase 3
        t2 = 0
        t3 = 0

        # Initialize the frequency of the best solution proposed
        global frequency_best_solution = 1

        # Initialise the cache to speed up calculation
        global cache = Dict{Tuple{Tuple, Tuple}, Tuple{OrderedDict, OrderedDict, Any}}()

        # Build models for all scenarios
        model_data_per_scenario = Dict{Int, Any}()
        model_data_per_scenario_PH = Dict{Int, Any}()
        for s in 1:NumScen
            model_data_per_scenario[s] = build_third_level_model(current_location, current_fortification, Arc_empty, data[:PStart], data[:CapHosp], data[:Dist], data[:Dis], data[:IwH], data[:Hosp], data[:I], TMax, Cap_Veh, s)
            
            model_data_per_scenario_PH[s] = build_models_PH(current_location, current_fortification, BudgetFortification, BudgetLocation, data[:CapLoc],
                Arc_empty, data[:Dist], data[:Dis], data[:IwH], data[:Hosp], data[:I], TMax, data[:PStart], Cap_Veh, data[:CapHosp], 
                Params_algorithm["Penalty"]["Params"]["WeightDevTrain"], Params_algorithm["Penalty"]["Params"]["WeightDevArc"], s)
            
        end

        # Create shortest paths
        g, ntoi, iton = dicts_to_weighted_graph(data[:Dist], data[:I])
        shortest_paths = all_pairs_shortest_paths(g, data[:I], ntoi, iton)

        # Calculate the arc values
        node_connections = get_edge_value(data[:I], data[:Dist])

        while !stopping

            t2 += @elapsed begin

                global key_prev_sol = (dict_to_tuple(current_location), dict_to_tuple(current_fortification))

                # Check if there is a solution in the cache already
                if !haskey(cache, key_prev_sol)

                    Threads.@threads for s in 1:NumScen

                        # Step 3a: Solve second stage

                        if Params_algorithm["SecondStage"]["Method"] == "SetCovering"

                            Arc_empty_ll = flatten_inner_dict_lowest_level(Arc_empty)
                            result_sp = level_2_ccg_scenarios(current_location, current_fortification, Arc_empty_ll, BudgetAttack, data[:I], data[:Dis], data[:IwH], data[:Hosp], data[:Dist], data[:PStart], data[:CapHosp], TMax, s, model_data_per_scenario, 
                                g, ntoi, iton, shortest_paths, node_connections, Params_algorithm["SecondStage"]["Params"]["Beta"])
                            results_sp_dict[s] = result_sp
                            
                        end

                        # Step 3b: Based on the results of the second stage, solve the third stage
                        if Params_algorithm["Penalty"]["Method"] == ""
                            update_RT_constraints_PH(model_data_per_scenario_PH[s], current_location)
                            update_FortifiedArc_constraints_PH(model_data_per_scenario_PH[s], current_fortification, results_sp_dict[s].attacked_arcs, BudgetLocation)
                            result_th_level = solve_model_PH(model_data_per_scenario_PH[s])
                            results_th_stage_dict[s] = result_th_level

                        end
                    end
                end
            end

            t3 += @elapsed begin

                # Check 
                if !haskey(cache, key_prev_sol)
                    # Step 3: Calculate the objective of the second stage based on the fixed locations (to evaluate quality of solution)
                    result_lower_stage = retrieve_obj(results_sp_dict, data[:ScenProb], "Weighted")
                    results_storage[counter] = OrderedDict("Loc" => current_location, "Fort" => current_fortification, "Obj" => result_lower_stage)

                    if result_lower_stage > best_obj
                        best_obj = result_lower_stage
                        best_location = deepcopy(current_location)
                        best_fortification = deepcopy(current_fortification)
                        frequency_best_solution = 1
                    elseif result_lower_stage == best_obj
                        frequency_best_solution += 1
                    end

                    # Step 4: Calculate the new suggested rt and arcs
                    # Step 4a: Calculate the new suggested rt
                    avg_suggested_rt = calculate_weighted_average_rt(results_th_stage_dict, data[:ScenProb])

                    # Step 4b: Calculate the new suggested arcs
                    avg_suggested_arcs = calculate_weighted_average_fortified(results_th_stage_dict, data[:ScenProb], empty_weighted_sums)

                    # Append the values to the cache
                    cache[key_prev_sol] = (avg_suggested_rt, avg_suggested_arcs, result_lower_stage)
                
                else 
                    # println("Nice, not necessary to solve again")
                    avg_suggested_rt, avg_suggested_arcs, current_obj = cache[key_prev_sol]
                    avg_suggested_rt_rep, avg_suggested_arcs_rep, result_lower_stage = cache[key_prev_sol]
                    results_storage[counter] = OrderedDict("Loc" => current_location, "Fort" => current_fortification, "Obj" => current_obj)

                    if current_obj == best_obj
                        frequency_best_solution += 1
                    end
                end

                if Params_algorithm["Voting"]["Method"] == "Top"
                    updated_locations = select_top_locations(avg_suggested_rt, data[:CapLoc], BudgetLocation)
                elseif Params_algorithm["Voting"]["Method"] == "Random"
                    updated_locations = select_top_locations_random(avg_suggested_rt, BudgetLocation, data[:CapLoc], Params_algorithm["Voting"]["Params"]["Level"])
                elseif Params_algorithm["Voting"]["Method"] == "Sigmoid"
                    updated_locations = select_top_locations_sigmoid(avg_suggested_rt, BudgetLocation, data[:CapLoc], Params_algorithm["Voting"]["Params"]["Randomness"], Params_algorithm["Voting"]["Params"]["Lambda"])
                elseif Params_algorithm["Voting"]["Method"] == "Hybrid"
                    updated_locations = select_top_fortified_locations_hybrid(avg_suggested_rt, data[:CapLoc], BudgetLocation, Params_algorithm["Voting"]["Params"]["Preselection"], Params_algorithm["Voting"]["Params"]["Level"])
                elseif Params_algorithm["Voting"]["Method"] == "EMA"
                    if counter == 1
                        global mvg_avg_loc = Dict()
                    end
                    updated_locations, mvg_avg_loc = select_top_locations_with_ema(avg_suggested_rt, BudgetLocation, data[:CapLoc], Params_algorithm["Voting"]["Params"]["Randomness"], Params_algorithm["Voting"]["Params"]["Alpha"], counter, mvg_avg_loc)
                end

                current_location = convert_top_locations(updated_locations, data[:I])

                if Params_algorithm["Voting"]["Method"] == "Top"
                    updated_arcs = select_top_fortified_arcs(avg_suggested_arcs, data[:Dist], BudgetFortification, data[:Hosp])
                elseif Params_algorithm["Voting"]["Method"] == "Random"
                    updated_arcs = select_top_fortified_arcs_random(avg_suggested_arcs, data[:Dist], BudgetFortification, data[:Hosp], Params_algorithm["Voting"]["Params"]["Level"])
                elseif Params_algorithm["Voting"]["Method"] == "Sigmoid"
                    updated_arcs = select_top_fortified_arcs_sigmoid(avg_suggested_arcs, data[:Dist], BudgetFortification, data[:Hosp], Params_algorithm["Voting"]["Params"]["Randomness"], Params_algorithm["Voting"]["Params"]["Lambda"])
                elseif Params_algorithm["Voting"]["Method"] == "Hybrid"
                    updated_arcs = select_top_fortified_arcs_hybrid(avg_suggested_arcs, data[:Dist], BudgetFortification, data[:Hosp], Params_algorithm["Voting"]["Params"]["Preselection"], Params_algorithm["Voting"]["Params"]["Level"])
                elseif Params_algorithm["Voting"]["Method"] == "EMA"
                    if counter == 1
                        global mvg_avg_arc = Dict()
                    end
                    #println(mvg_avg_arc)
                    updated_arcs, mvg_avg_arc = select_top_fortified_arcs_with_ema(avg_suggested_arcs, data[:Dist], BudgetFortification, data[:Hosp], Params_algorithm["Voting"]["Params"]["Randomness"], Params_algorithm["Voting"]["Params"]["Alpha"], counter, mvg_avg_arc)
                end
                
                current_fortification = convert_top_arcs(updated_arcs, data[:I])  

            end


            println("---------------")
            println("Counter ", counter)

            counter += 1
            # Check if ending criterion is met
            if Params_algorithm["Stopping"]["Method"] == "MaxIter"
                global stopping = check_max_iter(counter, Params_algorithm["Stopping"]["Params"]["NumIterations"])
            elseif Params_algorithm["Stopping"]["Method"] == "Max_proposals_best_solution"
                global stopping = current_best_repeatedly(best_location, best_fortification, frequency_best_solution, current_location, current_fortification, 
                    Params_algorithm["Stopping"]["Params"]["NumProposals"], counter, Params_algorithm["Stopping"]["Params"]["NumIterations"])
            elseif Params_algorithm["Stopping"]["Method"] == "Percentage_upper_bound"
                global stopping = percentage_upper_bound(result_lower_stage, Params_algorithm["Stopping"]["Params"]["UB"], Params_algorithm["Stopping"]["Params"]["Percentage"], 
                    counter, Params_algorithm["Stopping"]["Params"]["NumIterations"])
            end
        end


    end
    return (t1 = t1, 
            t2 = t2, 
            t3 = t3, 
            relaxed_sol = relaxed_solution,
            results = results_storage)

end