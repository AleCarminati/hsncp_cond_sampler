using LaTeXStrings, Plots, Serialization, StatsBase, StatsPlots

nclusHSNCP1 = deserialize("nclusHSNCP1.jls")
nclusHSNCP2 = deserialize("nclusHSNCP2.jls")
nclusHSNCP3 = deserialize("nclusHSNCP3.jls")
nclusHDP = deserialize("nclusHDP.jls")
ari1 = deserialize("ariHDPexperiment1.jls")
ari2 = deserialize("ariHDPexperiment2.jls")
ari3 = deserialize("ariHDPexperiment3.jls")

sdkvec = LinRange(0, 2, 8)

for i in eachindex(sdkvec)
  print(round.(mean(ari1[1][i, :]), digits = 3))
  print("\n")
end

print("\n")
for i in eachindex(sdkvec)
  print(round.(mean(ari2[1][i, :]), digits = 3))
  print("\n")
end

print("\n")
for i in eachindex(sdkvec)
  print(round.(mean(ari3[1][i, :]), digits = 3))
  print("\n")
end

arguments = Dict(
  :size => (639.64*0.65, 639.64*0.65/1.62), # Use golden ratio to compute height
  :legend => true,
  :titlefontsize => 12,
  :legendfontsize => 10,
  :tickfontsize => 8,
  :fontfamily => "Computer Modern",
  :grid => false,
  :xlabel => L"\sqrt{\sigma^2_T}",
)

plot(; arguments...)

plot!(sdkvec, mean(nclusHDP, dims = 2), marker = :circle, label = "HDP")

plot!(
  sdkvec,
  mean(nclusHSNCP1[1], dims = 2),
  marker = :circle,
  label = "HSNCP-1",
)
plot!(
  sdkvec,
  mean(nclusHSNCP2[1], dims = 2),
  marker = :circle,
  label = "HSNCP-2",
)
plot!(
  sdkvec,
  mean(nclusHSNCP3[1], dims = 2),
  marker = :circle,
  label = "HSNCP-3",
)

savefig("simulation_2.pdf")
