include("../src/import.jl")
using Serialization, Clustering

for ngroup in [50, 500]
  seed = 894975
  Random.seed!(seed)

  ngroup = 50

  # Generate dataset
  g = 2
  n = [ngroup, ngroup]

  means = [[-4-0.4, 4-0.45], [-4+0.4, 4+0.45]]
  components = [Normal.(means[l], 1) for l = 1:g]
  weights = [[1/2, 1/2] for l = 1:g]
  trueclust =
    [vcat([fill(1, Int64(ngroup/2)), fill(2, Int64(ngroup/2))]...) for l = 1:g]
  data = [rand.(components[l][trueclust[l]]) for l = 1:g]
  input = MCMCInput(data)

  model = NormalMeanVarVarModel(
    mixtshape = 3/2,
    mixtscale = 8.25,
    childrenprocess = GammaProcess(1.076),
    motherprocess = GammaProcess(1),
    motherlocmean = 0,
    motherlocsd = sqrt(10),
    motherlocshape = 1.658,
    motherlocscale = 0.529,
    nmotherprocesses = 1,
    dirparam = 1,
  )

  xgrid = LinRange(minimum(vcat(data...)), maximum(vcat(data...)), 500)

  output, predHSNCP = hsncpmixturemodel_fit(
    input,
    model;
    grid = xgrid,
    iterations = 5000,
    burnin = 1000,
    thin = 1,
  )

  # Concat the sampled across-group clustering labels in each group.
  clusHSNCP = hcat(output.agroupcluslabels...)

  @rput clusHSNCP

  R"library(salso)"
  R"bestclust = salso(clusHSNCP)"

  @rget bestclust

  bestclustHSNCP = deepcopy(bestclust)

  locshape = 1.71
  locscale = 15.81
  loclambda = 2

  @rput data
  @rput xgrid
  @rput locshape
  @rput locscale
  @rput loclambda
  @rput ngroup

  R"library(salso)"
  R"set.seed(43242)"
  R"output = hdp::HDPMarginalSampler(5000,1000,2,c(ngroup,ngroup),data,0,locshape,locscale,loclambda,1,1,1,1,1,2,FALSE,TRUE)"
  R"clusHDP = output$Partition"
  R"bestclust = salso(output$Partition)"
  R"predg1HDP = hdp::predictive(1, xgrid, output, 0,locshape,locscale,loclambda)"
  R"predg2HDP = hdp::predictive(2, xgrid, output, 0,locshape,locscale,loclambda)"

  @rget bestclust
  @rget predg1HDP
  @rget predg2HDP
  @rget clusHDP

  bestclustHDP = deepcopy(bestclust)

  truedens = [pdf.(components[l], xgrid')' * weights[l] for l = 1:g]

  l = @layout [grid(2, 1) grid(2, 1){0.3w}]
  arguments = Dict(
    :size => (639.64, 639.64/1.62), # Use golden ratio to compute height
    :layout => l,
    :legend => [:top false false false],
    :titlefontsize => 12,
    :legendfontsize => 10,
    :tickfontsize => 8,
    :fontfamily => "Computer Modern",
    :grid => false,
    :title => ["Group 1" "Group 2" "HSNCP" "HDP"],
  )

  plot(; arguments...)
  plot!(xgrid, truedens; linestyle = :dash, label = "True", subplot = [1 2])
  plot!(xgrid, predg1HDP[2, :], subplot = 1, label = "HDP")
  plot!(xgrid, predg2HDP[2, :], subplot = 2, label = "HDP")
  plot!(xgrid, mean(predHSNCP[1], dims = 1)', subplot = 1, label = "HSNCP")
  plot!(xgrid, mean(predHSNCP[2], dims = 1)', subplot = 2, label = "HSNCP")
  vline!(means[1], subplot = 1, linestyle = :dash, c = :grey, label = nothing)
  vline!(means[2], subplot = 2, linestyle = :dash, c = :grey, label = nothing)

  # First, order the samples in each group by their values.
  sort1 = vcat(sortperm(data[1]), sortperm(data[2]) .+ n[1])
  # Then, order the samples according to their true clustering.
  sort2 = sortperm(vcat(trueclust...))

  clusHSNCPordered = clusHSNCP[:, sort2]

  clusHDPordered = clusHDP[:, sort2]

  ntot = sum(n)

  psmHDP = zeros(ntot, ntot)
  psmHSNCP = zeros(ntot, ntot)

  for i = 1:ntot
    for j = 1:(i-1)
      psmHSNCP[i, j] =
        sum(clusHSNCPordered[:, i] .== clusHSNCPordered[:, j])/5000
      psmHDP[i, j] = sum(clusHDPordered[:, i] .== clusHDPordered[:, j])/5000
      psmHSNCP[j, i] = psmHSNCP[i, j]
      psmHDP[j, i] = psmHDP[i, j]
    end
    psmHSNCP[i, i] = 1
    psmHDP[i, i] = 1
  end

  heatmap!(
    psmHSNCP,
    color = :Blues,
    clim = (0, 1),
    colorbar = false,
    subplot = 3,
  )
  heatmap!(psmHDP, color = :Blues, clim = (0, 1), colorbar = false, subplot = 4)

  savefig("simulation_1_$ngroup.pdf")
end
