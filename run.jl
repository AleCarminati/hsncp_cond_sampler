include("import.jl")

# Set the seed for reproducibility of the experiments.
seed = 34545545454
Random.seed!(seed)

g = 3
n = [500, 500, 500]
components = [SkewNormal.([-3, 3], [1.5, 1.5], [20, 20]) for l = 1:g]
weights = [[0.5, 0.5] for l = 1:g]
trueclust = [rand(Categorical(weights[l]), n[l]) for l = 1:g]
data = [rand.(components[l][trueclust[l]]) for l = 1:g]
input = MCMCInput(data)
predictiongrid = LinRange(
  minimum([minimum(input.data[l]) for l = 1:input.g]),
  maximum([maximum(input.data[l]) for l = 1:input.g]),
  1000,
)
truedens = [
  hcat(map(x -> pdf.(x, predictiongrid), components[l])...) * weights[l] for
  l = 1:g
]

model = NormalMeanVarVarModel(
  mixtshape = 301,
  mixtscale = 0.5,
  childrenprocess = GammaProcess(1),
  motherprocess = GammaProcess(1),
  motherlocmean = 0,
  motherlocsd = 6,
  motherlocshape = 6,
  motherlocscale = 0.5,
  nmotherprocesses = 1,
  dirparam = 1,
)

output, prediction = hsncpmixturemodel_fit(
  input,
  model;
  grid = predictiongrid,
  iterations = 3000,
  burnin = 2000,
  thin = 1,
)

# Concat the sampled across-group clustering labels in each group.
clus = hcat(output.agroupcluslabels...)

@rput clus

R"library(salso)"
R"bestclust = salso(clus)"

@rget bestclust
