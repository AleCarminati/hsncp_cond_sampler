include("../src/import.jl")
using Serialization, Clustering

# Set the seed for reproducibility of the experiments.
seed = 56565656
Random.seed!(seed)

ndatasets = 100

sdkvec = LinRange(0, 2, 8)
nclusHDP = zeros(length(sdkvec), ndatasets)
nclusHSNCP = [zeros(length(sdkvec), ndatasets) for _ = 1:1]
ari = [zeros(length(sdkvec), ndatasets) for _ = 1:3]

for (idx, sdk) in enumerate(sdkvec)
  data = [[zeros(500), zeros(500)] for _ = 1:ndatasets]

  for it = 1:ndatasets
    # Generate dataset
    g = 2
    n = [500, 500]
    means = [
      [rand(Normal(-8, sdk)), rand(Normal(0, sdk)), rand(Normal(8, sdk))],
      [rand(Normal(-8, sdk)), rand(Normal(0, sdk)), rand(Normal(8, sdk))],
    ]
    components = [Normal.(means[l], 1) for l = 1:g]
    weights = [[1/3, 1/3, 1/3] for l = 1:g]
    trueclust = [rand(Categorical(weights[l]), n[l]) for l = 1:g]
    data[it] = [rand.(components[l][trueclust[l]]) for l = 1:g]
  end

  clus = [zeros(5000, 1000) for _ = 1:ndatasets]
  Threads.@threads for it = 1:ndatasets
    input = MCMCInput(data[it])

    vara = 4.52
    varb = 1.341

    locshape = 3/2
    locscale = 20.97

    model = NormalMeanVarVarModel(
      mixtshape = locshape,
      mixtscale = locscale,
      childrenprocess = GeneralizedGammaProcess(1.7, 0.05, 1),
      motherprocess = GammaProcess(5),
      motherlocmean = 0,
      motherlocsd = sqrt(10),
      motherlocshape = vara,
      motherlocscale = varb,
      nmotherprocesses = 1,
      dirparam = 1,
    )

    output, prediction = hsncpmixturemodel_fit(
      input,
      model;
      iterations = 5000,
      burnin = 1000,
      thin = 1,
    )

    # Concat the sampled across-group clustering labels in each group.
    clus[it] .= hcat(output.agroupcluslabels...)
  end

  for it = 1:ndatasets
    c = clus[it]

    @rput c

    R"library(salso)"
    R"bestclust = salso(c)"

    @rget bestclust

    nclusHSNCP[1][idx, it] = maximum(bestclust)
    ari[1][idx, it] =
      randindex(assignments(kmeans(vcat(data[it]...)', 3)), bestclust)[1]
  end
end

serialize("nclusHSNCP3.jls", nclusHSNCP)
serialize("ariHDPexperiment3.jls", ari)
