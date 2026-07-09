"""
This file contains the function to read the data
"""


using CSV, DataFrames, OrderedCollections, DelimitedFiles

#####################################################################################################################################################

function read_set(str_direct, name, del = ';')
    path = string(str_direct, name)
    data = readdlm(path, del)
    set = data[2:end, 1]
    return set
end

function read_2_d_dict(str_direct, name, del = ';')
    path = string(str_direct, name)
    df = CSV.read(path, DataFrame; header=1)  # Adjust the filename as needed

    # Convert DataFrame to OrderedDict
    row_labels = df[:, 1]
    col_labels = names(df)[2:end]

    col_labels = [tryparse(Int, col) !== nothing ? tryparse(Int, col) : col for col in col_labels]

    data = OrderedDict()

    for i in eachindex(row_labels)
        row_name = row_labels[i]
        data[row_name] = OrderedDict(col_labels[j] => df[i, j+1] for j in eachindex(col_labels))
    end
    return data
end

function import_instance(directory)
    string_directory = string(directory,"/")

    # Locations
    Dis = read_set(string_directory, "disaster_sites.csv")
    TS = read_set(string_directory, "train_stat.csv")
    Hosp = read_set(string_directory, "hospitals.csv")
    IwH = union(Dis, TS)
    I = union(Dis, TS, Hosp)

    # Distances
    Dist = read_2_d_dict(string_directory, "distances.csv")

    # Scenarios
    ScenProb = read_set(string_directory, "scenarios_prob.csv")
    PStart = read_2_d_dict(string_directory, "pat_start.csv")

    # Capacities
    path = string(string_directory, "loc_capacity.csv")
    CapLoc = CSV.File(path, header=true) |>Dict
    CapHosp = read_2_d_dict(string_directory, "hosp_capacity.csv")

    return (Dis = Dis, 
            TS = TS, 
            Hosp = Hosp, 
            IwH = IwH, 
            I = I, 
            Dist = Dist, 
            ScenProb = ScenProb, 
            PStart = PStart, 
            CapLoc = CapLoc, 
            CapHosp = CapHosp)
end