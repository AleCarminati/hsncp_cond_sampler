include("../src/import.jl")
using CSV, DataFrames, Serialization

ndatasets = 30

sumpred = CSV.read("sumpred.csv", DataFrame, header = false)
sumpred = Matrix(sumpred)

ari = deserialize("ari.jls")
print(mean(ari, dims = 2))

l = @layout [grid(1, 1){0.33h}; grid(2, 2)]
titles =
  ["Unimodal symmetric" "Multimodal" "Skewed - Converging" "Skewed - Same Direction" "Skewed - Diverging"]

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

arguments = Dict(
  :size => (639.64, 639.64),#1.62), # Use golden ratio to compute height
  :layout => l,
  :legend => [true false false false false],
  :titlefontsize => 12,
  :legendfontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :title => titles,
)

plot(; arguments...)

# Plot the true densities
for idx = 1:length(weights)
  plot!(
    xgrids[idx],
    (weights[idx]' * pdf.(distributions[idx], xgrids[idx]'))',
    subplot = idx,
    label = "True",
    linestyle = :dash,
  )
end

# Plot the grey lines for the mixture components composed of more than one
# Gaussian.
plot!(
  xgrids[1],
  (weights[1][1:5] .* pdf.(distributions[1][1:5], xgrids[1]'))',
  subplot = 1,
  color = :grey,
  label = nothing,
  linestyle = :dash,
)
plot!(
  xgrids[2],
  (weights[2] .* pdf.(distributions[2], xgrids[2]'))',
  subplot = 2,
  color = :grey,
  label = nothing,
  linestyle = :dash,
)

for expidx = 1:length(weights)
  plot!(
    xgrids[expidx],
    sumpred[expidx, :] ./ ndatasets,
    label = "HSNCP",
    subplot = expidx,
    color = 2,
  )
end

savefig("simulation_4_densities.pdf")
