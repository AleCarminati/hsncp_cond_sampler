using Serialization, StatsBase, Plots, LaTeXStrings

nclus = deserialize("nclus.jls")
nclusmean = mean(nclus, dims = 2)

varsa = [16.26, 14.39, 12.21, 10.25, 8.607, 7.269, 6.181, 5.305]
varsb = [25.03, 23.65, 21.16, 18.55, 16.12, 13.95, 12.05, 10.42]

arguments = Dict(
  :size => (639.64*0.65, 639.64*0.65/1.62), # Use golden ratio to compute height
  :legend => false,
  :guidefontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :xlabel => L"Expected value of $\sigma_j$",
  :ylabel => "Mean number of clusters",
  :marker => :circle,
  :color => :grey,
)

plot(sqrt.(varsb ./ (varsa .- 1)), nclusmean[1:end]; arguments...)

savefig("simulation_3.pdf")
