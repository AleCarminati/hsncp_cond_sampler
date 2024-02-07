include("sampler.jl")

# Set the seed for reproducibility of the experiments.
seed = 23478747247
Random.seed!(seed)

g = 3
data = [randn(rand(1:5)) for l = 1:g]
input = MCMCInput(data)

output = hsncpmixturemodel_fit(
  input;
  dimchildrenloc = 1,
  iterations = 5,
  burnin = 5,
  thin = 1,
)
print(output)
