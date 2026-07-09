#####################################################################################################################################################

# This file checks whether the ending criterion is met

#####################################################################################################################################################

function check_max_iter(counter, max_iter)
    if counter >= max_iter
        return true
    else
        return false
    end
end

function check_equality_nodes(nodes_1, nodes_2)
    return sort(nodes_1) == sort(nodes_2)
end

function check_equality_arcs(arcs_1, arcs_2)
    return arcs_1 == arcs_2
end

function current_best_repeatedly(best_nodes, best_arcs, frequency, current_nodes, current_arcs, max_proposals, counter, max_iter)
    """
    This function stops if the current best solution has been proposed repeatedly
    """
    equal_nodes = check_equality_nodes(current_nodes, best_nodes)
    equal_arcs = check_equality_arcs(current_arcs, best_arcs)
    if equal_nodes && equal_arcs && frequency > max_proposals
        return true
    else
        if counter >= max_iter
            return true
        else
            return false
        end
    end
end

function percentage_upper_bound(current_obj, ub, percentage, counter, max_iter)
    if current_obj >= ub * percentage
        return true
    else
        if counter >= max_iter
            return true
        else
            return false
        end
    end
end