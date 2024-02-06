include("sampler.jl")

# Set the seed for reproducibility of the experiments.
seed = 23478747247
Random.seed!(seed)

g = 3
n = rand(1:5, g)
data = randn(sum(n))
input = MCMCInput(data, n, g)

output = hsncpmixturemodel_fit(input; iterations = 5, dimchildrenloc = 1)
print(output)