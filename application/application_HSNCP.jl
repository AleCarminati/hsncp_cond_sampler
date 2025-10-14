include("import.jl")
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

model = NormalMeanVarVarModel(
  mixtshape = 3/2,
  mixtscale = 1/2*(0.235-0.6265/(12.0867-1)),
  childrenprocess = GeneralizedGammaProcess(1.306, 0.05, 1),
  motherprocess = GeneralizedGammaProcess(5, 0.1, 0.2),
  motherlocmean = 1.989,
  motherlocsd = 1,
  motherlocshape = 12.0867,
  motherlocscale = 0.6265,
  nmotherprocesses = 1,
  dirparam = 1,
)

output, prediction = hsncpmixturemodel_fit(
  input,
  model;
  grid = predictiongrid,
  iterations = 5000,
  burnin = 10000,
  thin = 10,
  fkthreshold = 1e-6,
)

# Concat the sampled across-group clustering labels in each group.
clus = hcat(output.agroupcluslabels...)

@rput clus

R"library(salso)"
R"bestclus = salso(clus)"

@rget bestclus

meanpred = zeros(input.g+1, 1000)
for group = 1:(input.g+1)
  meanpred[group, :] = mean(prediction[group], dims = 1)
end

serialize("output.jls", output)
serialize("meanprediction.jls", meanpred)
serialize("bestclus.jls", bestclus)
