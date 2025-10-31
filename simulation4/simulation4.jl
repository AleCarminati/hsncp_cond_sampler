include("../src/import.jl")
using Serialization, Clustering, CSV, Tables

varsa = [2.427, 1.726, 1.165, 3.501, 7.765]
varsb = [1.203, 0.6918, 0.1734, 1.741, 2.445]

nobs = 3000

xgrids = [
  LinRange(-15, 15, 1000),
  LinRange(-8, 8, 1000),
  LinRange(-6, 6, 1000),
  LinRange(-3.5, 9, 1000),
  LinRange(-6, 6, 1000),
]

distributions = [
  [
    Normal(-12, 0.3),
    Normal(-12, 2),
    Normal(-5.5, 1),
    Normal(-4, 1),
    Normal(-2.5, 1),
    Normal(4, 1),
    TDist(3)+12,
  ],
  [
    Normal(-6, 0.3),
    Normal(-5, 0.3),
    Normal(-4, 0.3),
    Normal(-3, 0.3),
    Normal(6, 0.3),
    Normal(5, 0.3),
    Normal(4, 0.3),
    Normal(3, 0.3),
  ],
  [SkewNormal(-5.5, 1.5, 20), SkewNormal(5.5, 1.5, -20)],
  [SkewNormal(-3, 2, 20), SkewNormal(3, 2, 20)],
  [SkewNormal(-1, 2, -20), SkewNormal(1, 2, 20)],
]

weights = [
  [1/8, 1/8, 1/12, 1/12, 1/12, 1/4, 1/4],
  [1.5/16, 2.5/16, 2.5/16, 1.5/16, 2.5/16, 1.5/16, 1.5/16, 2.5/16],
  [1/2, 1/2],
  [1/2, 1/2],
  [1/2, 1/2],
]

cluspdf = [
  [
    x->weights[1][1:2]'*pdf.(distributions[1][1:2], x),
    x->weights[1][3:5]'*pdf.(distributions[1][3:5], x),
    x->weights[1][6]*pdf(distributions[1][6], x),
    x->weights[1][7]*pdf(distributions[1][7], x),
  ],
  [
    x->weights[2][1:4]'*pdf.(distributions[2][1:4], x),
    x->weights[2][5:8]'*pdf.(distributions[2][5:8], x),
  ],
  [
    x->weights[3][1]*pdf(distributions[3][1], x),
    x->weights[3][2]*pdf(distributions[3][2], x),
  ],
  [
    x->weights[4][1]*pdf(distributions[4][1], x),
    x->weights[4][2]*pdf(distributions[4][2], x),
  ],
  [
    x->weights[5][1]*pdf(distributions[5][1], x),
    x->weights[5][2]*pdf(distributions[5][2], x),
  ],
]

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
  215884,
  2253435,
  2365445,
  24656,
  2544,
  26344,
  276949,
  2854543,
  29544,
  30123,
]

ari = zeros(length(xgrids), length(seeds))

sumpred = zeros(length(weights), length(xgrids[1]))

for expidx = 1:length(weights)
  clus = [zeros(5000, nobs) for _ = 1:length(seeds)]
  trueclus = [zeros(nobs) for _ = 1:length(seeds)]

  Threads.@threads for idx = 1:length(seeds)
    Random.seed!(seeds[idx])

    # Generate the dataset
    data = zeros(nobs)
    categories = rand(Categorical(weights[expidx]), nobs)
    data = rand.(distributions[expidx][categories])

    datapdfs = zeros(length(cluspdf[expidx]), nobs)
    for clusidx = 1:length(cluspdf[expidx])
      datapdfs[clusidx, :] = cluspdf[expidx][clusidx].(data)
    end

    trueclus[idx] .= map(argmax, eachcol(datapdfs))

    input = MCMCInput([data])

    model = NormalMeanVarVarModel(
      mixtshape = 3/2,
      mixtscale = 1/2*(var(data)-varsb[expidx]/(varsa[expidx]-1)),
      childrenprocess = GammaProcess(1),
      motherprocess = GammaProcess(5),
      motherlocmean = 0,
      motherlocsd = sqrt(64),
      motherlocshape = varsa[expidx],
      motherlocscale = varsb[expidx],
      nmotherprocesses = 1,
      dirparam = 1,
    )

    output, pred = hsncpmixturemodel_fit(
      input,
      model;
      grid = xgrids[expidx],
      iterations = 5000,
      burnin = 1000,
      thin = 1,
    )

    clus[idx] .= hcat(output.agroupcluslabels...)
    sumpred[expidx, :] += mean(pred[1], dims = 1)'
  end

  for idx = 1:length(seeds)
    c = clus[idx]

    @rput c

    R"library(salso)"
    R"bestclus = salso(c)"

    @rget bestclus

    ari[expidx, idx] = randindex(Int64.(trueclus[idx]), bestclus)[1]
  end
end

CSV.write("sumpred.csv", Tables.table(sumpred), writeheader = false)

serialize("ari.jls", ari)
