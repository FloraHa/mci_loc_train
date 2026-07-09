"""
This file stores the additional functions
"""

using OrderedCollections

#####################################################################################################################################################

function get_valid_arcs(Dist)
    valid_arcs = []
    for i in keys(Dist)
        for j in keys(Dist[i])
            if Dist[i][j] < 1000
                push!(valid_arcs, (i, j))
            end
        end
    end
    return valid_arcs
end

function get_invalid_arcs(Dist)
    invalid_arcs = []
    for i in keys(Dist)
        for j in keys(Dist[i])
            if Dist[i][j] >= 1000
                push!(invalid_arcs, (i, j))
            else
                if i == j
                    push!(invalid_arcs, (i, j))
                end
            end
        end
    end
    return invalid_arcs
end

function extend_dict(ArcMP::OrderedDict, num::Int)
    for origin in keys(ArcMP)
        for destination in keys(ArcMP[origin])
            for n in 2:num
                ArcMP[origin][destination][n] = 0
            end
        end
    end
    return ArcMP
end

function create_ArcMP(I)
    ArcMP = OrderedDict{String, OrderedDict{String, OrderedDict{Int64, Int64}}}()
    for i in I
        ArcMP[i] = OrderedDict{String, OrderedDict{Int64, Int64}}()
        for j in I
            ArcMP[i][j] = OrderedDict{Int64, Int64}(1 => 0)
        end
    end
    return ArcMP
end

function flatten_inner_dict_scenarios(outer_dict::OrderedDict{String, OrderedDict{String, OrderedDict{Int64, Int64}}})
    result = OrderedDict{String, OrderedDict{String, OrderedDict{Int64, Int64}}}()
    for (key, inner_dict) in outer_dict
        inner_result = OrderedDict{String, OrderedDict{Int64, Int64}}()
        for (inner_key, value_dict) in inner_dict
            flattened_inner = OrderedDict{Int64, Int64}()
            for (value_key, value_inner_dict) in value_dict
                flattened_inner[value_key] = first(values(value_inner_dict))
            end
            inner_result[inner_key] = flattened_inner
        end
        result[key] = inner_result
    end

    return result
end

function create_arc_values(selected_arcs, I)
    arc_values = OrderedDict{String, OrderedDict{String, Int64}}()

    for i in I
        arc_values[i] = OrderedDict{String, Int64}()
        for j in I
            if i == j
                arc_values[i][j] = 0  # No self-loops
            else
                if get(selected_arcs, (i, j), 0.0) > 0 || get(selected_arcs, (j, i), 0.0) > 0
                    arc_values[i][j] = 1
                else
                    arc_values[i][j] = 0
                end
            end
        end
    end

    return arc_values
end

function flatten_inner_dict_lowest_level(outer_dict::OrderedDict{String, OrderedDict{String, OrderedDict{Int64, Int64}}})
    result = OrderedDict{String, OrderedDict{String, Int64}}()
    for (key, inner_dict) in outer_dict
        inner_result = OrderedDict{String, Int64}()
        for (inner_key, value_dict) in inner_dict
            value = first(values(value_dict))
            inner_result[inner_key] = value
        end
        result[key] = inner_result
    end
    return result
end

function get_most_used_arcs(vsm, I)
    frequency = Dict()
        for i in I
            for j in I
                if !haskey(frequency, (i,j))
                    frequency[(i,j)] = 0
                    frequency[(j,i)] = 0
                end
            frequency[(i,j)] += vsm[i,j]
            frequency[(j,i)] += vsm[i,j]
            end
        end
    return frequency
end

function filter_used_arcs(frequency)
    used_arcs = Dict()
    for arcs in keys(frequency)
        if frequency[arcs] > 0
            used_arcs[arcs] = frequency[arcs]
        end
    end
    return used_arcs
end

function merge_arc_counts(used_arcs)
    merged_arcs = Dict{Tuple{String, String}, Float64}()  # Use String instead of Int

    for (i, j) in keys(used_arcs)
        arc = Tuple(sort(collect((i, j))))  # Sort the arc lexicographically

        # Accumulate the values
        merged_arcs[arc] = get(merged_arcs, arc, 0.0) + used_arcs[(i, j)]
    end

    return merged_arcs
end

function select_top_arcs(merged_arcs, budget::Int)
    sorted_arcs = sort(collect(merged_arcs), by = x -> -x[2])  # x[2] is the value

    selected_arcs = Dict{Tuple{String, String}, Float64}()

    for (i, (arc, value)) in enumerate(sorted_arcs)
        if i > budget/2  # Stop if the budget is exhausted
            break
        end
        selected_arcs[arc] = value
    end

    return selected_arcs
end

function create_arc_values(selected_arcs, I)
    arc_values = OrderedDict{String, OrderedDict{String, Int64}}()

    for i in I
        arc_values[i] = OrderedDict{String, Int64}()
        for j in I
            if i == j
                arc_values[i][j] = 0  # No self-loops
            else
                if get(selected_arcs, (i, j), 0.0) > 0 || get(selected_arcs, (j, i), 0.0) > 0
                    arc_values[i][j] = 1
                else
                    arc_values[i][j] = 0
                end
            end
        end
    end

    return arc_values
end

function create_empty_dict_weighted_sums(data)
    empty_weighted_sums = Dict{String,Dict{String,Float64}}()

        for loc in keys(data[:Dist])
            if loc in data[:Hosp]
                continue
            end
            subdict = Dict{String,Float64}()
            for sub_loc in keys(data[:Dist][loc])
                if sub_loc in data[:Hosp]
                    continue
                end

                d = data[:Dist][loc][sub_loc]
                if 0 < d < 1000
                    subdict[sub_loc] = 0.0
                end
            end
            if !isempty(subdict)
                empty_weighted_sums[loc] = subdict
            end
        end
    return empty_weighted_sums
end

###################################################################################################
# Additional functions for the second level
###################################################################################################

function get_most_used_arcs_with_t(vsm, I, TMax)
    frequency = Dict()
    for t in 1:TMax
        for i in I
            for j in I
                if t == 1
                    if !haskey(frequency, (i,j))
                        frequency[(i,j)] = vsm[i,j,t]
                        frequency[(j,i)] = vsm[i,j,t]
                    end
                end
                frequency[(i,j)] += vsm[i,j,t]
                frequency[(j,i)] += vsm[i,j,t]
            end
        end
    end
    return frequency
end

function normalise(unique_arcs)
    normalised_arcs = Dict()
    for (i, j) in keys(unique_arcs)
        if unique_arcs[(i, j)] >= 1
            normalised_arcs[(i, j)] = 1
        end
    end
    return normalised_arcs
end

function filter_used_arcs(frequency)
    used_arcs = Dict()
    for arcs in keys(frequency)
        if frequency[arcs] > 0
            used_arcs[arcs] = frequency[arcs]
        end
    end
    return used_arcs
end

###################################################################################################
# Additional functions for printing
###################################################################################################

function retrieve_obj(results_sp_dict, probability_scen, mode)
    if mode == "Min"
        relevant_value = Inf
        for (key, inner_dict) in results_sp_dict
            if inner_dict.obj < relevant_value  # Directly access 'obj'
                relevant_value = inner_dict.obj
            end
        end
    elseif mode == "Weighted"
        relevant_value = sum(inner_dict.obj * probability_scen[key] for (key, inner_dict) in results_sp_dict)
    end
    return relevant_value
end

function print_locations(dict)
    for (key, value) in dict
        if value > 0
            println("$key => $value")
        end
    end
end

function dict_to_tuple(d)
    return tuple((k => v for (k, v) in d)...)
end

function print_fortified_arcs_scenarios(attacked_arcs)
    for (from_node, to_nodes) in attacked_arcs
        for (to_node, value) in to_nodes
            if value > 0
                println("From $from_node to $to_node: $value")
            end
        end
    end
end