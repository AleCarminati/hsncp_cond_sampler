# This file contains auxiliary plotting functions.

function plotclustering(
  input::MCMCInput,
  trueclust,
  bestclust,
  filename;
  distancetrueest = 1,
  distancegroups = 0.25,
)
  #= Creates a scatter plot comparing estimated and true clustering.
  		Optional arguments:
  		- distancetrueest: represents the y-axis distance between the points colored
  			based on the true clustering and the points colored based on the estimated
  			clustering.
  		- distancegroups: represents the y-axis distance between points in the same
  			group.
  	=#
  ticksvalues = vcat(
    0:distancegroups:((input.g-1)*distancegroups),
    (distancetrueest+(input.g-1)*distancegroups):distancegroups:(distancetrueest+(input.g-1)*2*distancegroups),
  )
  tickslabels = vcat(
    "Estimated - group " .* string.(1:input.g),
    "True - group " .* string.(1:input.g),
  )

  yvalues = vcat(fill.(ticksvalues[1:input.g], n)...)
  yvalues = vcat(yvalues, fill.(ticksvalues[input.g+1:2*input.g], n)...)

  savefig(
    scatter(
      repeat(vcat(data...), 2),
      yvalues,
      color = vcat(bestclust, vcat(trueclust...)),
      yticks = (ticksvalues, tickslabels),
      legend = false,
      size = (1500, 700),
      title = "Across-group clustering",
    ),
    filename,
  )
end

function plotdensitypredictions(
  input::MCMCInput,
  predictiongrid,
  truedens,
  prediction,
  filename,
)
  #= Creates a grid of plots:
    - For each group, it plots the predictive density for a new data point in
      that group, the density that generated data in that group and an histogram
      of data in that group.
    - It plots the predictive density for a new data point in a new group.
    =#

  plots = []

  for l = 1:(input.g)
    push!(
      plots,
      histogram(
        input.data[l],
        normalize = :probability,
        color = :lightblue,
        label = nothing,
        bins = 12,
      ),
    )
    plot!(
      predictiongrid,
      [
        truedens[l],
        vec(sum(prediction[l], dims = 1)) ./ size(prediction[l])[1],
      ],
      title = "Group $l",
      size = (1500, 700),
      label = ["True density" "Predicted density"],
      lw = 2,
    )
  end

  push!(
    plots,
    plot(
      predictiongrid,
      vec(sum(prediction[input.g+1], dims = 1)) ./
      size(prediction[input.g+1])[1],
      title = "Predictive density for a new observation in group $(input.g+1)",
      size = (1500, 700),
      legend = false,
    ),
  )

  savefig(plot(plots...), filename)
end

function atomsvalues(atomsvector, model::GammaCRMModel; value = "jump")
  #= Helper function that, given a vector of AtomsContainer, returns, for each
    atom, a vector containing its values. If the atom is not allocated in that
    a certain iteration, it returns the "missing" value. =#

  niterations = size(atomsvector)[1]
  vecmaxatoms = map(x -> size(x.jumps)[1], atomsvector)
  natoms = maximum(vecmaxatoms)

  if value == "jump"
    values = map(
      y -> map(
        x -> y <= vecmaxatoms[x] ? atomsvector[x].jumps[y] : missing,
        1:niterations,
      ),
      1:natoms,
    )
  elseif value == "counter"
    values = map(
      y -> map(
        x -> y <= vecmaxatoms[x] ? atomsvector[x].counter[y] : missing,
        1:niterations,
      ),
      1:natoms,
    )
  elseif value == "mean"
    values = map(
      y -> map(
        x -> y <= vecmaxatoms[x] ? atomsvector[x].locations[y][1] : missing,
        1:niterations,
      ),
      1:natoms,
    )
  elseif value == "var"
    values = map(
      y -> map(
        x -> y <= vecmaxatoms[x] ? atomsvector[x].locations[y][2] : missing,
        1:niterations,
      ),
      1:natoms,
    )
  else
    error("Invalid value: '$input_string'")
  end

  return values
end

function plottrace(values, filename; iterations = nothing)
  #= Creates a traceplot. To have multiple traces, provide a vector of vectors
    in input.
    Optional arguments:
    - Iterations: a vector of vectors that, for each trace, represents the
      iterations where that trace must be plotted.
  =#

  savefig(
    plot(iterations, values, size = (1500, 700), legend = false),
    filename,
  )
end
