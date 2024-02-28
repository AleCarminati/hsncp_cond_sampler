include("import.jl")

# Set the seed for reproducibility of the experiments.
seed = 23478747247
Random.seed!(seed)

g = 3
n = [50, 50, 50]
components = Normal.([-5, 0, 5], [1, 1, 1])
trueclust = [rand(1:3, n[l]) for l = 1:g]
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

ticksvalues = vcat(0:0.5:((g-1)*0.5), ((g+1)*0.5):0.5:g)
tickslabels =
  vcat("Estimated - group " .* string.(1:g), "True - group " .* string.(1:g))

yvalues = vcat([fill((l - 1) * 0.5, n[l]) for l = 1:g]...)
yvalues =
  vcat(yvalues, [fill(((g + 1) * 0.5) + (l - 1) * 0.5, n[l]) for l = 1:g]...)

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
