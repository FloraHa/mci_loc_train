using Graphs
using SimpleWeightedGraphs
using DataStructures

function dicts_to_weighted_graph(Dist, nodes; absent_weight=1000)
    n = length(nodes)

    node_to_idx = Dict{String,Int}()
    idx_to_node = Vector{String}(undef, n)

    for (i, name) in enumerate(nodes)
        node_to_idx[name] = i
        idx_to_node[i] = name
    end

    g = SimpleWeightedGraph(n)

    for u in nodes
        ui = node_to_idx[u]
        for (v, w) in Dist[u]
            if w < absent_weight && haskey(node_to_idx, v)
                vi = node_to_idx[v]
                if ui != vi
                    add_edge!(g, ui, vi, float(w))
                end
            end
        end
    end

    return g, node_to_idx, idx_to_node
end

function all_pairs_shortest_paths(g, nodes, node_to_idx, idx_to_node)
    results = Dict{String,Tuple{Vector{Float64},Vector{Int}}}()

    for s in nodes
        si = node_to_idx[s]
        sp = dijkstra_shortest_paths(g, si)
        results[s] = (sp.dists, sp.parents)
    end

    return results
end

function scenario_distances(results, node_to_idx, RT, Dis, Hosp, PStart, CapHosp, s)

    active_disasters = filter(d -> PStart[d][s] > 0, Dis)
    available_hospitals = filter(h -> CapHosp[h][s] > 0, Hosp)
    train_locations = filter(t -> get(RT, t, 0) > 0, keys(RT))
    
    train_to_dis = Dict{Tuple{String,String}, Float64}()
    for t in train_locations
        dist_vec, _ = results[t]
        for d in active_disasters
            d_idx = node_to_idx[d]
            train_to_dis[(t,d)] = dist_vec[d_idx] * PStart[d][s]
        end
    end

    dis_to_hosp = Dict{Tuple{String,String}, Float64}()
    for d in active_disasters
        dist_vec, _ = results[d]
        for h in available_hospitals
            h_idx = node_to_idx[h]
            dis_to_hosp[(d,h)] = dist_vec[h_idx] * PStart[d][s] * CapHosp[h][s]
        end
    end

    return train_to_dis, dis_to_hosp
end

function min_train_to_disaster(train_to_dis, d)
    vals = [v for ((_, dis), v) in train_to_dis if dis == d]
    return isempty(vals) ? 1000 : minimum(vals)
end

# Minimum time from disaster site `d` to any hospital
function min_disaster_to_hospital(dis_to_hosp, d)
    vals = [v for ((dis, _), v) in dis_to_hosp if dis == d]
    return isempty(vals) ? 1000 : minimum(vals)
end

function graph_without_arcs(g, interdicted_arcs)
    g2 = SimpleWeightedGraph(copy(adjacency_matrix(g)))
    
    for (u,v) in interdicted_arcs
        if has_edge(g2, u, v)
            rem_edge!(g2, u, v)
        end
        if has_edge(g2, v, u)
            rem_edge!(g2, v, u)
        end
    end

    return g2
end

function sum_min_travel_times(g, node_to_idx,
                            RT, Dis, Hosp, PStart, CapHosp, s, 
                              interdicted_arcs)

    g_mod = isempty(interdicted_arcs) ? g : graph_without_arcs(g, interdicted_arcs)
    active_disasters = filter(d -> PStart[d][s] > 0, Dis)
    available_hospitals = filter(h -> CapHosp[h][s] > 0, Hosp)
    train_locations = filter(t -> get(RT, t, 0) > 0, keys(RT))

    train_dist_maps = Dict{String, Vector{Float64}}()
    for t in train_locations
        si = node_to_idx[t]
        sp = dijkstra_shortest_paths(g_mod, si)
        train_dist_maps[t] = sp.dists
    end

    disaster_dist_maps = Dict{String, Vector{Float64}}()
    for d in active_disasters
        di = node_to_idx[d]
        sp = dijkstra_shortest_paths(g_mod, di)
        disaster_dist_maps[d] = sp.dists
    end

    total_train_time = 0.0
    total_hosp_time = 0.0

    for d in active_disasters
        train_times = [train_dist_maps[t][node_to_idx[d]] * PStart[d][s] for t in train_locations]
        min_train_time = minimum(train_times)
        if min_train_time == Inf
            min_train_time = 1000
        end

        disaster_times = [disaster_dist_maps[d][node_to_idx[h]] * PStart[d][s] * CapHosp[h][s] for h in available_hospitals]
        min_hosp_time = minimum(disaster_times)
        if min_hosp_time == Inf
            min_hosp_time = 1000
        end

        total_train_time += min_train_time
        total_hosp_time += min_hosp_time
    end

    return total_train_time, total_hosp_time
end
