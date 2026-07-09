"""
This file is the main file for the algorithm
"""

using JSON3

include("Data_reading.jl")
include("Add_functions.jl")
include("Algorithm.jl")
include("Graph_generation_V2.jl")

###################################################################################################

# Parameter setting

###################################################################################################


directory = "Datafiles/6n_7e_10m_3d_3s_42os_2ss_5ps"
NumScens = 3
BudgetLocation = 1
BudgetFortification = 2
BudgetAttack = 1

data = import_instance(directory)

Cap_Veh = 1

ArcMP = create_ArcMP(data[:I])
ArcMP = extend_dict(ArcMP, NumScens)

TMax = 30
maxiter = 3 * NumScens

Params_algorithm = Dict("Initialisation" => Dict("Method" => "Simple", "Params" => Dict()),
                        "Stopping" => Dict("Method" => "Max_proposals_best_solution", "Params" => Dict("NumProposals" => 3, "NumIterations" => maxiter)),
                        "SecondStage" => Dict("Method" => "SetCovering", "Params" => Dict("Beta" => 1.0)), 
                        "Penalty" => Dict("Method" => "", "Params" => Dict("WeightDevTrain" => 0.25, "WeightDevArc" => 1.0)),
                        "Voting" => Dict("Method" => "Hybrid", "Params" => Dict("Preselection" => 0.5, "Level" => 1.0)))


reps = 1
println("=============================")

for rep in 1:reps
    results = main_algorithm(BudgetLocation, BudgetFortification, BudgetAttack,
        ArcMP, data, Params_algorithm, TMax, NumScens, Cap_Veh)


    max_counter = argmax(x -> x[2]["Obj"], results.results |> collect) |> first
    for counter in keys(results.results)
        println("Objective $counter: ", results.results[counter]["Obj"])
    end

end

