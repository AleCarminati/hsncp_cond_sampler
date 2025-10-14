include("import.jl")
using Serialization, CSV, DataFrames, KernelDensity

df = CSV.read("galaxy_labels.csv", DataFrame, drop = [1])
rename!(df, :Column2 => :urdiff)

# Group observations by cl_lum and cl_den. The first grouping index is cl_lum,
# the second grouping index is cl_den.
# Example: data[8] will contain all urdiff values for cl_lum=2 and cl_den=3.
data = [
  df.urdiff[df.cl_lum .== i.&&df.cl_den .== j] for
  (i, j) in Iterators.product(1:5, 1:5)
]
data = reshape(permutedims(data), (25,))

predictiongrid = LinRange(minimum(df.urdiff), maximum(df.urdiff), 1000)

bestclus = deserialize("bestclus.jls")
meanpred = deserialize("meanprediction.jls")
bestclusHDP = deserialize("bestclusHDP.jls")
meanpredHDP = deserialize("meanpredictionHDP.jls")

flatdata = vcat(data...)

output = deserialize("output.jls")
nit = size(output.mixtparams, 1)

n = length(flatdata)

# --- HSNCP density and clustering plot ---

arguments = Dict(
  :size => (639.64, 639.64*1.25),
  :titlefontsize => 12,
  :legendfontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :yticks => false,
  :legend => false,
  :plot_title => "HSNCP",
  :layout => (2, 1),
)

plot(; arguments...)

for (i, j) in Iterators.product(1:5, 1:5)
  plot!(predictiongrid, meanpred[(i-1)*5+j, :], subplot = 1)
end

yval = vcat([repeat([i], length(data[i])) for i = 1:25]...)
scatter!(flatdata, yval, color = bestclus, subplot = 2)

savefig("plots/sloan_HSNCP.pdf")

# --- HDP density and clustering plot ---

arguments = Dict(
  :size => (639.64, 639.64*1.25),
  :titlefontsize => 12,
  :legendfontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :yticks => false,
  :legend => false,
  :plot_title => "HDP",
  :layout => (2, 1),
)

plot(; arguments...)

for (i, j) in Iterators.product(1:5, 1:5)
  plot!(predictiongrid, meanpredHDP[(i-1)*5+j, :], subplot = 1)
end

yval = vcat([repeat([i], length(data[i])) for i = 1:25]...)
scatter!(flatdata, yval, color = bestclusHDP, subplot = 2)

savefig("sloan_HDP.pdf")

# --- Stacked barplot ---

arguments = Dict(
  :size => (639.64, 639.64*0.5),
  :titlefontsize => 12,
  :legendfontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :yticks => false,
  :legend => false,
  :palette => :tab20,
  :xticks => permutedims([
    (1:maximum(bestclus), ["Cluster $group" for group = 1:maximum(bestclus)]),
    (
      1:maximum(bestclusHDP),
      ["Cluster $group" for group = 1:maximum(bestclusHDP)],
    ),
  ]),
  :xrotation => 45,
  :yticks => [0.0, 1.0],
  :layout => (1, 2),
  :title => ["HSNCP" "HDP"],
  :bottom_margin => (5, :mm),
)

plot(; arguments...)

indexes = cumsum(length.(data))

cumpercentage = repeat([1.0], maximum(bestclus))
cluscount = counts(bestclus)

bar!(1:maximum(bestclus), cumpercentage, subplot = 1)

for i = 25:-1:1
  cumpercentage .-=
    counts(
      bestclus[(indexes[i]-length(data[i])+1):indexes[i]],
      maximum(bestclus),
    ) ./ cluscount
  bar!(1:maximum(bestclus), cumpercentage, subplot = 1)
end

cumpercentage = repeat([1.0], maximum(bestclusHDP))
cluscount = counts(bestclusHDP)

bar!(1:maximum(bestclusHDP), cumpercentage, subplot = 2)

for i = 25:-1:1
  cumpercentage .-=
    counts(
      bestclusHDP[(indexes[i]-length(data[i])+1):indexes[i]],
      maximum(bestclusHDP),
    ) ./ cluscount
  bar!(1:maximum(bestclusHDP), cumpercentage, subplot = 2)
end

savefig("sloan_clus_barplot.pdf")

# --- Entropy plot ---

indexes = length.(data)

group = vcat([fill(i, indexes[i]) for i in eachindex(indexes)]...)

entrHDP = zeros(maximum(bestclusHDP))
entrHSNCP = zeros(maximum(bestclus))

for i = 1:maximum(bestclusHDP)
  probs = counts(group[bestclusHDP .== i], 25) ./ sum(bestclusHDP .== i)
  entrHDP[i] = entropy(probs)
end

for i = 1:maximum(bestclus)
  probs = counts(group[bestclus .== i], 25) ./ sum(bestclus .== i)
  entrHSNCP[i] = entropy(probs)
end

maxentropy = round.(entropy(repeat([1/25], 25)), sigdigits = 3)

arguments = Dict(
  :size => (639.64*0.75, 639.64*0.75*0.5),
  :titlefontsize => 12,
  :axisfontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :yticks => false,
  :legend => false,
  :yticks => permutedims([[0, maxentropy/2, maxentropy], []]),
  :layout => (1, 2),
  :title => ["HSNCP" "HDP"],
  :xlabel => "Cluster",
)

plot(; arguments...)

bar!(1:maximum(bestclus), entrHSNCP, color = :grey70, subplot = 1)

bar!(1:maximum(bestclusHDP), entrHDP, color = :grey70, subplot = 2)

hline!([maxentropy], linestyle = :dash, color = :grey, subplot = 1)
hline!([maxentropy], linestyle = :dash, color = :grey, subplot = 2)

savefig("sloan_entropy.pdf")

# --- Save dataset for alluvial plot ---
indexes = length.(data)

group = vcat([fill(i, indexes[i]) for i in eachindex(indexes)]...)

df = DataFrame(bestclus = bestclus, bestclusHDP = bestclusHDP, group = group)

CSV.write("alluvialdf.csv", df)
