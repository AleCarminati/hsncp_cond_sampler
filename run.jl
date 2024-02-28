include("import.jl")

# Set the seed for reproducibility of the experiments.
Random.seed!(seed)

g = 3
components = [Normal.([-5, 0, 5], [1, 1, 1]) for l = 1:g]
components = [SkewNormal.([-5, 0, 5], [4, 4, 4], [50, 50, 50]) for l = 1:g]
trueclust = [rand(1:3, n[l]) for l = 1:g]
data = [rand.(components[l][trueclust[l]]) for l = 1:g]
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

distancetrueest = 1
interval = 0.25
ticksvalues = vcat(
  0:interval:((g-1)*interval),
  (distancetrueest+(g-1)*interval):interval:(distancetrueest+(g-1)*2*interval),
)
tickslabels =
  vcat("Estimated - group " .* string.(1:g), "True - group " .* string.(1:g))

yvalues = vcat(fill.(ticksvalues[1:g], n)...)
yvalues = vcat(yvalues, fill.(ticksvalues[g+1:2*g], n)...)

savefig(
  scatter(
    repeat(vcat(data...), 2),
    yvalues,
    color = vcat(bestclust, vcat(trueclust...)),
    yticks = (ticksvalues, tickslabels),
    legend = false,
    size = (1500, 700),
    title = "Across-group clustering",
  ),
  "clust.png",
)
