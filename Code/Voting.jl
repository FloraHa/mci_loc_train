#####################################################################################################################################################

# This file contains the functions to perform the voting procedure

#####################################################################################################################################################

using StatsBase
using Base.Threads

#####################################################################################################################################################

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

function calculate_weighted_average_rt(results_dict, scenario_probs)

    first_scenario = first(values(results_dict)).rt_new_val
    weighted_sums = OrderedDict(loc => 0.0 for loc in keys(first_scenario))

    for (scenario, result_obj) in results_dict
        prob = scenario_probs[scenario]
        rt_new_val = result_obj.rt_new_val
        rt_importance = result_obj.departure_counts

        for (loc, val) in rt_new_val
            if haskey(rt_importance, loc)
                weighted_sums[loc] += val * prob * (rt_importance[loc]+1)
            else
                weighted_sums[loc] += val * prob * 0
            end
        end
    end

    return weighted_sums
end

function calculate_weighted_average_fortified(results_dict, scenario_probs, empty_dict_weighted_sums)

    weighted_sums = deepcopy(empty_dict_weighted_sums)

    for (scenario, result_obj) in results_dict
        prob = scenario_probs[scenario]
        fortified_new_val = result_obj.fortified_new_val

        for (loc, sub_dict) in fortified_new_val
            if haskey(weighted_sums, loc)
                for (sub_loc, val) in sub_dict
                    if haskey(weighted_sums[loc], sub_loc)
                        weighted_sums[loc][sub_loc] += val * prob
                    end
                end
            end
        end
    end

    return weighted_sums
end

function select_top_locations(weighted_sums, CapLoc, BudgetLocations)

    valid_locations = filter(loc -> CapLoc[loc[1]] > 0, collect(weighted_sums))

    sorted_locations = sort(valid_locations, by = loc -> loc[2], rev = true)

    top_locations = sorted_locations[1:min(BudgetLocations, length(sorted_locations))]

    top_location_keys = [loc[1] for loc in top_locations]

    return top_location_keys
end

function select_top_locations_random(weighted_sums, BudgetLocations, CapLoc, randomness_level::Float64=0.5)

    filtered_locations = filter(loc -> CapLoc[loc[1]] > 0, collect(weighted_sums))
    sorted_locations = sort(collect(filtered_locations), by = x -> x[2], rev = true)
    
    location_keys = [loc[1] for loc in sorted_locations]
    location_weights = [loc[2] for loc in sorted_locations]
    
    total_weight = sum(location_weights)
    probabilities = location_weights / total_weight

    if randomness_level == 0.0
        selected_locations = location_keys[1:BudgetLocations]
    elseif randomness_level == 1.0
        selected_locations = sample(location_keys, BudgetLocations, replace=false)
    else
        uniform_prob = fill(1.0 / length(location_keys), length(location_keys))
        mixed_probabilities = (1 - randomness_level) * probabilities .+ randomness_level * uniform_prob
        selected_locations = sample(location_keys, Weights(mixed_probabilities), BudgetLocations)
    end

    return selected_locations
end

function select_top_locations_sigmoid(weighted_sums, BudgetLocations, CapLoc, randomness_level::Float64=0.5, lambda_factor::Float64=5.0)

    filtered_locations = filter(loc -> CapLoc[loc[1]] > 0, collect(weighted_sums))
    sorted_locations = sort(collect(filtered_locations), by = x -> x[2], rev = true)

    location_keys = [loc[1] for loc in sorted_locations]
    location_weights = [loc[2] for loc in sorted_locations]
    
    total_weight = sum(location_weights)
    probabilities = location_weights / total_weight

    exp_weights = exp.(lambda_factor * probabilities)
    soft_probabilities = exp_weights / sum(exp_weights)

    if randomness_level == 0.0
        selected_locations = location_keys[1:BudgetLocations]
    elseif randomness_level == 1.0
        selected_locations = sample(location_keys, BudgetLocations, replace=false)
    else
        uniform_prob = fill(1.0 / length(location_keys), length(location_keys))
        mixed_probabilities = (1 - randomness_level) * soft_probabilities .+ randomness_level * uniform_prob

        selected_locations = sample(location_keys, Weights(mixed_probabilities), BudgetLocations, replace=false)
    end

    return selected_locations
end

function select_top_fortified_locations_hybrid(weighted_sums, CapLoc, BudgetLocation, top_percent::Float64=0.3, randomness_level::Float64=0.5)

    filtered_locations = filter(loc -> CapLoc[loc[1]] > 0, collect(weighted_sums))
    sorted_locations = sort(collect(filtered_locations), by=x -> x[2], rev=true)

    locations_keys = [loc[1] for loc in sorted_locations]
    locations_weights = [loc[2] for loc in sorted_locations]

    top_count = max(1, round(Int, top_percent * length(locations_keys)), BudgetLocation)
    top_locations_subset = locations_keys[1:top_count]

    subset_weights = [weighted_sums[loc] for loc in top_locations_subset]
    total_weight = sum(subset_weights)
    probabilities = subset_weights / total_weight

    uniform_prob = fill(1.0 / length(top_locations_subset), length(top_locations_subset))
    mixed_probabilities = (1 - randomness_level) * probabilities .+ randomness_level * uniform_prob

    selected_locations = sample(top_locations_subset, Weights(mixed_probabilities), min(BudgetLocation, length(top_locations_subset)), replace=false)

    return selected_locations
end

function select_top_locations_with_ema(weighted_sums, BudgetLocations, CapLoc, randomness_level, alpha, iteration, moving_avgs)

    filtered_locations = filter(loc -> CapLoc[loc[1]] > 0, collect(weighted_sums))
    sorted_locations = sort(filtered_locations, by = x -> x[2], rev = true)

    location_keys = [loc[1] for loc in sorted_locations]
    location_weights = [loc[2] for loc in sorted_locations]

    for (loc, weight) in zip(location_keys, location_weights)
        if iteration == 1
            moving_avgs[loc] = weight
        else
            moving_avgs[loc] = alpha * weight + (1 - alpha) * get(moving_avgs, loc, weight)
        end
    end

    total_weight = sum(values(moving_avgs))
    probabilities = Dict(loc => moving_avgs[loc] / total_weight for loc in keys(moving_avgs) if total_weight > 0)

    if randomness_level == 0.0
        selected_locations = location_keys[1:min(BudgetLocations, length(location_keys))]
    elseif randomness_level == 1.0
        selected_locations = sample(location_keys, BudgetLocations, replace=false)
    else
        uniform_prob = fill(1.0 / length(location_keys), length(location_keys))
        mixed_probabilities = Dict(loc => (1 - randomness_level) * probabilities[loc] + randomness_level * uniform_prob[i] for (i, loc) in enumerate(location_keys))

        selected_locations = sample(location_keys, Weights([mixed_probabilities[loc] for loc in location_keys]), BudgetLocations)
    end

    return selected_locations, moving_avgs
end

function select_top_fortified_arcs(weighted_sums, distance, BudgetFortification, Hosp)
    arcs_with_sums = []

    for (loc, sub_dict) in weighted_sums
        for (sub_loc, sum_val) in sub_dict
            if !(loc in Hosp || sub_loc in Hosp) && loc != sub_loc && distance[loc][sub_loc] > 0 && distance[loc][sub_loc] < 1000
                push!(arcs_with_sums, ((loc, sub_loc), sum_val))
            end
        end
    end

    sorted_arcs = sort(arcs_with_sums, by = x -> x[2], rev = true)

    top_arcs = sorted_arcs[1:min(BudgetFortification * 2, length(sorted_arcs))]

    top_arc_pairs = []

    added_arcs = Dict()

    while length(added_arcs) < min(BudgetFortification * 2, length(sorted_arcs))

        (loc, sub_loc), _ = top_arcs[1]
            
        if !(get(added_arcs, (loc, sub_loc), false) || get(added_arcs, (sub_loc, loc), false))
            push!(top_arc_pairs, (loc, sub_loc))
            push!(top_arc_pairs, (sub_loc, loc))
            added_arcs[(loc, sub_loc)] = true
            added_arcs[(sub_loc, loc)] = true
        end

        top_arcs = filter(x -> x[1] != (loc, sub_loc) && x[1] != (sub_loc, loc), top_arcs)
    
    end

    return top_arc_pairs
end

function select_top_fortified_arcs_random(weighted_sums, distance, BudgetFortification, Hosp, randomness_level::Float64=0.5)
    arcs_with_sums = []

    for (loc, sub_dict) in weighted_sums
        for (sub_loc, sum_val) in sub_dict
            if distance[loc][sub_loc] > 0 && distance[loc][sub_loc] < 1000 &&
                !(loc in Hosp || sub_loc in Hosp)
                push!(arcs_with_sums, ((loc, sub_loc), sum_val))
            end
        end
    end

    sorted_arcs = sort(arcs_with_sums, by = x -> x[2], rev = true)

    arcs_keys = [arc[1] for arc in sorted_arcs]  # List of (loc, sub_loc)
    arcs_weights = [arc[2] for arc in sorted_arcs]  # List of corresponding weights

    total_weight = sum(arcs_weights)
    probabilities = arcs_weights / total_weight

    top_arc_pairs = Set()

    if randomness_level == 0.0
        top_arcs = arcs_keys[1:min(BudgetFortification, length(arcs_keys))]
    elseif randomness_level == 1.0
        while length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
            arc = sample(arcs_keys, 1, replace=false)[1]
            if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
                push!(top_arc_pairs, arc)
                push!(top_arc_pairs, (arc[2], arc[1]))
            end
        end

    else
        uniform_prob = fill(1.0 / length(arcs_keys), length(arcs_keys))
        mixed_probabilities = (1 - randomness_level) * probabilities .+ randomness_level * uniform_prob

        while length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
            arc = sample(arcs_keys, Weights(mixed_probabilities), 1, replace=false)[1]
            if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
                push!(top_arc_pairs, arc)
                push!(top_arc_pairs, (arc[2], arc[1]))
            end
        end

    end

    return top_arc_pairs
end

function select_top_fortified_arcs_sigmoid(weighted_sums, distance, BudgetFortification, Hosp, randomness_level::Float64=0.5, lambda_factor::Float64=5.0)

    arcs_with_sums = []

    for (loc, sub_dict) in weighted_sums
        for (sub_loc, sum_val) in sub_dict
            if distance[loc][sub_loc] > 0 && distance[loc][sub_loc] < 1000 &&
                !(loc in Hosp || sub_loc in Hosp)
                push!(arcs_with_sums, ((loc, sub_loc), sum_val))
            end
        end
    end

    sorted_arcs = sort(arcs_with_sums, by = x -> x[2], rev = true)

    arcs_keys = [arc[1] for arc in sorted_arcs]
    arcs_weights = [arc[2] for arc in sorted_arcs]

    total_weight = sum(arcs_weights)
    probabilities = arcs_weights / total_weight

    exp_weights = exp.(lambda_factor * probabilities)
    soft_probabilities = exp_weights / sum(exp_weights)

    top_arc_pairs = Set()

    if randomness_level == 0.0
        top_arcs = arcs_keys[1:min(BudgetFortification, length(arcs_keys))]
    elseif randomness_level == 1.0
        while length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
            arc = sample(arcs_keys, 1, replace=false)[1]
            if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
                push!(top_arc_pairs, arc)
                push!(top_arc_pairs, (arc[2], arc[1]))
            end
        end

    else
        uniform_prob = fill(1.0 / length(arcs_keys), length(arcs_keys))
        mixed_probabilities = (1 - randomness_level) * soft_probabilities .+ randomness_level * uniform_prob

        while length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
            arc = sample(arcs_keys, Weights(mixed_probabilities), 1, replace=false)[1]
            if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
                push!(top_arc_pairs, arc)
                push!(top_arc_pairs, (arc[2], arc[1]))
            end
        end
    end

    return top_arc_pairs
end

function remove_arc_and_reverse!(arc, arcs_list)
    reverse_arc = (arc[2], arc[1])
    filter!(x -> x != arc && x != reverse_arc, arcs_list)
end


function select_top_fortified_arcs_hybrid(weighted_sums, distance, BudgetFortification, Hosp, top_percent::Float64=0.3, randomness_level::Float64=0.5)
    arcs_with_sums = []
    for (loc, sub_dict) in weighted_sums
        for (sub_loc, sum_val) in sub_dict
            push!(arcs_with_sums, ((loc, sub_loc), sum_val))
        end
    end

    sorted_arcs = sort(arcs_with_sums, by = x -> x[2], rev = true)

    arcs_keys = [arc[1] for arc in sorted_arcs]  # List of (loc, sub_loc)
    arcs_weights = [arc[2] for arc in sorted_arcs]  # List of corresponding weights

    top_count = max(1, round(Int, top_percent * length(arcs_keys)), BudgetFortification*2)
    top_arcs_subset = arcs_keys[1:top_count]  # Top X% arcs
    top_arcs_subset = collect(top_arcs_subset)

    subset_weights = [weighted_sums[arc[1]][arc[2]] for arc in top_arcs_subset]
    total_weight = sum(subset_weights)
    if total_weight > 0
        normalized = subset_weights ./ total_weight
    else
        normalized = fill(1.0 / length(top_arcs_subset), length(top_arcs_subset))
    end

    mixed_probabilities =
        (1 - randomness_level) .* normalized .+
        randomness_level .* (1.0 ./ length(top_arcs_subset))

    top_arc_pairs = Set()

    while !isempty(top_arcs_subset) && length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
        arc = sample(top_arcs_subset, Weights(mixed_probabilities), 1, replace=false)[1]
        if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
            push!(top_arc_pairs, arc)
            push!(top_arc_pairs, (arc[2], arc[1]))
        end
        remove_arc_and_reverse!(arc, top_arcs_subset)
        subset_weights = [weighted_sums[a[1]][a[2]] for a in top_arcs_subset]
        total_weight = sum(subset_weights)
        if total_weight > 0
            mixed_probabilities = (1 - randomness_level) * (subset_weights ./ total_weight) .+ randomness_level * (1.0 ./ length(top_arcs_subset))
        else
            mixed_probabilities = fill(1.0 / length(top_arcs_subset), length(top_arcs_subset))
        end

    end
    # end

    return top_arc_pairs
end

function select_top_fortified_arcs_random(weighted_sums, distance, BudgetFortification, Hosp, randomness_level::Float64=0.5)
    arcs_with_sums = []
    for (loc, sub_dict) in weighted_sums
        for (sub_loc, sum_val) in sub_dict
            push!(arcs_with_sums, ((loc, sub_loc), sum_val))
        end
    end

    sorted_arcs = sort(arcs_with_sums, by = x -> x[2], rev = true)

    arcs_keys = [arc[1] for arc in sorted_arcs]  # List of (loc, sub_loc)
    arcs_weights = [arc[2] for arc in sorted_arcs]  # List of corresponding weights

    top_arcs_subset = collect(arcs_keys)
    subset_weights = [weighted_sums[arc[1]][arc[2]] for arc in top_arcs_subset]
    total_weight = sum(subset_weights)
    if total_weight > 0
        normalized = subset_weights ./ total_weight
    else
        normalized = fill(1.0 / length(top_arcs_subset), length(top_arcs_subset))
    end

    mixed_probabilities =
        (1 - randomness_level) .* normalized .+
        randomness_level .* (1.0 ./ length(top_arcs_subset))

    top_arc_pairs = Set()

    while !isempty(top_arcs_subset) && length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
        arc = sample(top_arcs_subset, Weights(mixed_probabilities), 1, replace=false)[1]
        if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
            push!(top_arc_pairs, arc)
            push!(top_arc_pairs, (arc[2], arc[1]))
        end
        remove_arc_and_reverse!(arc, top_arcs_subset)
        subset_weights = [weighted_sums[a[1]][a[2]] for a in top_arcs_subset]
        total_weight = sum(subset_weights)
        if total_weight > 0
            mixed_probabilities = (1 - randomness_level) * (subset_weights ./ total_weight) .+ randomness_level * (1.0 ./ length(top_arcs_subset))
        else
            mixed_probabilities = fill(1.0 / length(top_arcs_subset), length(top_arcs_subset))
        end

    end

    return top_arc_pairs
end

function select_top_fortified_arcs_with_ema(weighted_sums, distance, BudgetFortification, Hosp, randomness_level, alpha, iteration, moving_avgs)
    arcs_with_sums = []

    for (loc, sub_dict) in weighted_sums
        for (sub_loc, sum_val) in sub_dict
            push!(arcs_with_sums, ((loc, sub_loc), sum_val))
        end
    end

    sorted_arcs = sort(arcs_with_sums, by = x -> x[2], rev = true)

    arcs_keys = [arc[1] for arc in sorted_arcs]
    arcs_weights = [arc[2] for arc in sorted_arcs]

    if iteration == 1
        moving_avgs = Dict(arc => weight for (arc, weight) in arcs_with_sums)
    else
        arc_values = Dict(arc => weight for (arc, weight) in arcs_with_sums)
        moving_avgs_prev = deepcopy(moving_avgs)
        for arc in keys(arc_values)
            moving_avgs[arc] = alpha * arc_values[arc] + (1 - alpha) * moving_avgs_prev[arc]
        end
    end

    total_weight = sum(values(moving_avgs))
    if total_weight > 0
        probabilities = Dict(arc => moving_avgs[arc] / total_weight for arc in keys(moving_avgs))
    else
        probabilities = Dict(arc => 1.0 / length(moving_avgs) for arc in keys(moving_avgs))
    end
    top_arc_pairs = Set()

    uniform_prob = fill(1.0 / length(arcs_keys), length(arcs_keys))
    mixed_probabilities = Dict(
        arc => (1 - randomness_level) * probabilities[arc] + randomness_level * uniform_prob[i]
        for (i, arc) in enumerate(arcs_keys)
    )

    while length(top_arc_pairs) < min(2 * BudgetFortification, length(arcs_keys))
        arc = sample(arcs_keys, Weights([mixed_probabilities[arc] for arc in arcs_keys]), 1, replace=false)[1]
        if !(arc in top_arc_pairs || (arc[2], arc[1]) in top_arc_pairs)
            push!(top_arc_pairs, arc)
            push!(top_arc_pairs, (arc[2], arc[1]))
        end
    end

    return top_arc_pairs, moving_avgs
end

function convert_top_locations(top_locations, I)
    locations_dict = OrderedDict{String, Int64}()
    for loc in I
        if loc in top_locations
            locations_dict[loc] = 1
        else
            locations_dict[loc] = 0
        end
    end
    return locations_dict
end

function convert_top_arcs(top_arcs, I)
    arcs_dict = OrderedDict{String, OrderedDict{String, Int64}}()
    for loc in I
        arcs_dict[loc] = OrderedDict()
        for dest in I
            if (loc, dest) in top_arcs
                arcs_dict[loc][dest] = 1
            else
                arcs_dict[loc][dest] = 0
            end
        end
    end
    return arcs_dict
end