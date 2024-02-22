include("import.jl")

# Set the seed for reproducibility of the experiments.
seed = 23478747247
Random.seed!(seed)

g = 3
data = [randn(rand(1:5)) for l = 1:g]
input = MCMCInput(data)

model = NormalMeanModel(1, 1, 1, 1, 1)

output =
  hsncpmixturemodel_fit(input, model; iterations = 5, burnin = 5, thin = 1)
print(output)
