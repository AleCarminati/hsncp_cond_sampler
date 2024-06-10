# This file contains auxiliary plotting functions.

function plotclustering(
  input::MCMCInput,
  trueclust,
  bestclust,
  truedensclus,
  bestdensclus,
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
  availablemarkers = [
    :circle,
    :rect,
    :star5,
    :diamond,
    :hexagon,
    :cross,
    :xcross,
    :utriangle,
    :dtriangle,
    :pentagon,
    :star4,
    :star6,
    :+,
    :x,
  ]
  markervec = []
  for l = 1:input.g
    markervec = vcat(markervec, fill(availablemarkers[bestdensclus[l]], n[l]))
  end
  for l = 1:input.g
    markervec = vcat(markervec, fill(availablemarkers[truedensclus[l]], n[l]))
  end

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
      title = "Across-group and density clustering",
      markershape = markervec,
    ),
    filename,
  )
end

function plotgroupeddensitypredictions(
  input::MCMCInput,
  predictiongrid,
  prediction,
  bestdensclus,
  filename;
)
  #= For each density cluster plots the predictive density for a new
    data point in each group in that density cluster. =#

  plots = []

  for m in unique(bestdensclus)
    first = true
    for l in findall(bestdensclus .== m)
      if first
        push!(
          plots,
          plot(
            predictiongrid,
            vec(sum(prediction[l], dims = 1)) ./ size(prediction[l])[1],
            title = "Cluster $m",
            linewidth = 2,
            color = bestdensclus[l],
            size = (1500, 700),
            xlims = extrema(predictiongrid),
            legend = false,
          ),
        )
        first = false
      else
        plot!(
          predictiongrid,
          vec(sum(prediction[l], dims = 1)) ./ size(prediction[l])[1],
          linewidth = 2,
          color = bestdensclus[l],
        )
      end
    end
  end

  savefig(plot(plots...), filename)
end

function plotdensitypredictions(
  input::MCMCInput,
  predictiongrid,
  prediction,
  bestdensclus,
  filename;
  truedens = nothing,
  truedensclus = nothing,
  groupsidx = nothing,
  plotprednewgroup = true,
  layout = nothing,
)
  #= Creates a grid of plots:
    - For each group in groupsidx, it plots the predictive density for a new
      data point in that group, the density that generated data in that group
      and an histogram of data in that group.
    - It plots the predictive density for a new data point in a new group.

    # Optional Arguments
    - `truedens`: True densities for each group. If not provided, the true
      densities are not drawn.
    - `truedensclus`: Cluster assignments for the true densities. They must
      be provided if and only if `truedens` is provided.
    - `groupsidx`: Indices of the groups to plot (default is `nothing`,
      plots all groups).
    - `plotprednewgroup`: Boolean to plot the predictive density for a new group
      (default is `true`).
    - `layout`: Layout of the plot grid (default is the default layout set by
      the Plots package).
    =#

  plots = []

  if groupsidx == nothing
    groupsidx = 1:input.g
  end

  if layout == nothing
    layout = size(groupsidx)[1] + plotprednewgroup
  end

  for l in groupsidx
    push!(
      plots,
      histogram(
        input.data[l],
        normalize = :pdf,
        color = :lightblue,
        label = nothing,
        bins = 12,
        size = (1500, 700),
        xlims = extrema(predictiongrid),
      ),
    )
    if truedens != nothing
      plot!(
        predictiongrid,
        truedens[l],
        title = "Group $l",
        label = "True density",
        linewidth = 2,
        linestyle = :dot,
        color = truedensclus[l],
      )
    end
    plot!(
      predictiongrid,
      vec(sum(prediction[l], dims = 1)) ./ size(prediction[l])[1],
      label = "Predicted density",
      linewidth = 2,
      color = bestdensclus[l],
    )
  end

  if plotprednewgroup
    push!(
      plots,
      plot(
        predictiongrid,
        vec(sum(prediction[input.g+1], dims = 1)) ./
        size(prediction[input.g+1])[1],
        title = "Predictive density for a new observation in group $(input.g+1)",
        size = (1500, 700),
        legend = false,
        color = :black,
      ),
    )
  end

  savefig(plot(plots..., layout = layout), filename)
end

function atomsvalues(
  atomsvector,
  model::GammaCRMModel;
  value = "jump",
  iterations = nothing,
)
  #= Helper function that, given a vector of AtomsContainer, returns, for each
    atom, a vector containing its values. If the atom is not allocated in that
    a certain iteration, it returns the "missing" value. =#

  niterations = size(atomsvector)[1]
  vecmaxatoms = map(x -> size(x.jumps)[1], atomsvector)
  natoms = maximum(vecmaxatoms)

  # If the iterations are not specified, return the values in every iteration.
  if iterations == nothing
    iterations = 1:niterations
  end

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
  elseif value == "natoms"
    values = vecmaxatoms
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
