include("import.jl")

# Set the seed for reproducibility of the experiments.
seed = 23478747247
Random.seed!(seed)

g = 3
components = Normal.([-5, 0, 5], [1, 1, 1])
trueclust = [rand(1:3, 50) for l = 1:g]
data = [rand.(components[trueclust[l]]) for l = 1:g]
input = MCMCInput(data)

model = NormalMeanModel(1, 1, 1, 1, 1)

output = hsncpmixturemodel_fit(
  input,
  model;
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

p1 = scatter(vcat(data...), fill(2, 150), color = vcat(trueclust...))
p2 = scatter(vcat(data...), fill(1, 150), color = bestclust)

savefig(
  scatter(
    repeat(vcat(data...), 2),
    vcat(fill(1, 150), fill(2, 150)),
    color = vcat(bestclust, vcat(trueclust...)),
    yticks = ([1, 2], ["Estimated clustering", "True clustering"]),
    legend = false,
    size = (1500, 700),
  ),
  "clust.png",
)
