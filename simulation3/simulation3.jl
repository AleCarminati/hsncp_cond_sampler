include("import.jl")
using Serialization, StatsBase

varsa = [16.26, 14.39, 12.21, 10.25, 8.607, 7.269, 6.181, 5.305]
varsb = [25.03, 23.65, 21.16, 18.55, 16.12, 13.95, 12.05, 10.42]

nvars = length(varsa)

childmasses = [16.34, 19.16, 21.99, 25.52, 27.64, 29.75, 32.58, 33.99]

@assert length(varsa)==length(varsb)==length(childmasses)

seeds = [
  1832,
  256413,
  3823,
  43433,
  5666642,
  664545,
  765125,
  8455435,
  944332,
  1044523,
  114553,
  12454365,
  135463,
  145436,
  1563464,
  164364,
  17423,
  18542452,
  195433,
  205356,
  215453,
  224612,
  235434,
  24775,
  25534,
  2643552,
  2754353,
  2845,
  2911523,
  306645,
]
nclus = zeros(nvars, length(seeds))

for (idx, seed) in enumerate(seeds)
  Random.seed!(seed)

  # Generate the dataset
  data = [rand(Normal(2, 1), 700), rand(Normal(-2, 1), 700)]

  input = MCMCInput(data)

  clus = [zeros(5000, 1400) for _ = 1:nvars]

  Threads.@threads for varidx = 1:nvars
    model = NormalMeanVarVarModel(
      mixtshape = 3/2,
      mixtscale = 1/2*(var(vcat(data...))-varsb[varidx]/(varsa[varidx]-1)),
      childrenprocess = GammaProcess(childmasses[varidx]),
      motherprocess = GammaProcess(1/2),
      motherlocmean = 0,
      motherlocsd = sqrt(4),
      motherlocshape = varsa[varidx],
      motherlocscale = varsb[varidx],
      nmotherprocesses = 1,
      dirparam = 1,
    )

    output, _ = hsncpmixturemodel_fit(
      input,
      model;
      iterations = 5000,
      burnin = 1000,
      thin = 1,
    )

    clus[varidx] .= hcat(output.agroupcluslabels...)
  end

  for varidx = 1:nvars
    c = clus[varidx]
    @rput c

    R"library(salso)"
    R"bestclust = salso(c)"

    @rget bestclust

    nclus[varidx, idx] = maximum(bestclust)
  end
end

serialize("nclus.jls", nclus)
