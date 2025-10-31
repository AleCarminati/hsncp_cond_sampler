include("../src/import.jl")
using Serialization, CSV, DataFrames

# Set the seed for reproducibility of the experiments.
seed = 89777837
Random.seed!(seed)

df = CSV.read("galaxy_labels.csv", DataFrame, drop = [1])
rename!(df, :Column2 => :urdiff)

# Group observations by cl_lum and cl_den. The first grouping index is cl_lum,
# the second grouping index is cl_den.
# Example: data[8] will contain all urdiff values for cl_lum=2 and cl_den=3.
data = [
  df.urdiff[df.cl_lum .== i.&&df.cl_den .== j] for
  (i, j) in Iterators.product(1:5, 1:5)
]
data = reshape(permutedims(data), (25,))

input = MCMCInput(data)
predictiongrid = LinRange(
  minimum([minimum(input.data[l]) for l = 1:input.g]),
  maximum([maximum(input.data[l]) for l = 1:input.g]),
  1000,
)

seldata = input.data
nl = input.n
locshape = 2.331
locscale = 0.524
loclambda = 1/4
nit = 5000
thin = 10

@rput nit
@rput thin
@rput seldata
@rput predictiongrid
@rput nl
@rput locshape
@rput locscale
@rput loclambda

R"library(salso)"
R"set.seed(43242)"
R"output = hdp::HDPMarginalSampler(nit*thin,10000,length(nl),nl,seldata,1.989,locshape,locscale,loclambda,1,1,1,1,2,1,FALSE,TRUE)"
R"mus = output$mu"
R"sigmas = output$sigma"
R"clusHDP = output$Partition"
R"bestclus = salso(output$Partition[seq(from = 1, by = thin, length.out = nit),])"
R"predHDP = hdp::predictive_all_groups(predictiongrid, output, 1.989,locshape,locscale,loclambda)"

@rget bestclus
@rget clusHDP
@rget predHDP

# Apply thinning
clusHDP = Int64.(clusHDP[1:thin:end, :])

meanpred = zeros(input.g, 1000)
for group = 1:input.g
  meanpred[group, :] = predHDP[group][2, :]
end

serialize("clusHDP.jls", clusHDP)
serialize("meanpredictionHDP.jls", meanpred)
serialize("bestclusHDP.jls", bestclus)
