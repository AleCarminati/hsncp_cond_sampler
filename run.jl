include("sampler.jl")

# Set the seed for reproducibility of the experiments.
seed = 23478747247
Random.seed!(seed)

g = 3
n = rand(1:5, g)
data = randn(sum(n))
input = MCMCInput(data, n, g)

output = hsncpmixturemodel_fit(
  input;
  dimchildrenloc = 1,
  iterations = 5,
  burnin = 5,
  thin = 1,
)
print(output)
