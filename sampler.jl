# This file contains the functions to run the conditional sampler for the
# HSNCP mixture model.

function initalizemcmcstate!(state::MCMCState, model::NormalMeanModel)
  # TODO: initalization of the state of the MCMC.
end

function updatemotherprocess!(state::MCMCState, model::Model)
  # Update the allocated atoms of the mother process.
  updatemotherprocessalloc!(state, model)
  # Update the non allocated atoms of the mother process.
  updatemotherprocessnonalloc!(state, model)
end

function updatemotherprocessalloc!(state::MCMCState, model::NormalMeanModel)
  # First, sample the jumps.

  g = size(state.childrenallocatedatoms)[1]
  shape = model.motherjumpshape .+ state.motherallocatedatoms.counter
  rate =
    model.motherjumprate + g - sum(
      (model.childenjumprate ./ (state.auxu .+ model.childenjumprate)) .^
      (model.childenjumpshape),
    )
  # The rate is equal for all the Gamma distributions, therefore transform it
  # from a scalar to a vector with the same values.
  rate = fill(rate, size(shape)[1])

  # We write the inverse of the second parameter because the Gamma() function
  # requires the shape and the scale.
  gammas = Gamma.(shape, 1 ./ rate)

  state.motherallocatedatoms.jumps[:] = rand.(gammas, 1)

  # Then, sample the locations.

  standevs =
    sqrt.(
      1 ./ (
        1 / model.motherlocsd^2 .+
        state.motherallocatedatoms.counter ./ model.kernelsd^2
      )
    )

  # For each allocated atom j of the mother process, compute the sum of all the
  # atoms of the children processes that are associated with j.
  sums = map(
    x -> sum(
      map(
        y ->
          transpose(state.childrenatomslabels[y] .== x) *
          state.childrenallocatedatoms[y].jumps,
        Vector(1:g),
      ),
    ),
    1:size(state.motherallocatedatoms.jumps)[1],
  )

  means =
    sums ./ (model.kernelsd) .^ (2 .* state.motherallocatedatoms.counter) .*
    standevs

  normals = Normal.(means, standevs)

  state.motherallocatedatoms.locations[:] = rand.(normals, 1)
end

function updatemotherprocessnonalloc!(state::MCMCState, model::NormalMeanModel)
  g = size(state.childrenallocatedatoms)[1]

  new_rate =
    model.motherjumprate + g - sum(
      (model.childenjumprate ./ (state.auxu .+ model.childenjumprate)) .^
      model.childenjumpshape,
    )

  coef = (model.motherjumprate / new_rate)^model.motherjumpshape

  f =
    x ->
      coef * gamma(model.motherjumpshape, new_rate * x) /
      gamma(model.motherjumpshape)

  state.mothernonallocatedatoms.jumps = fergusonklass(f, 0.1)

  state.mothernonallocatedatoms.locations = rand(
    Normal(0, model.motherlocsd),
    size(state.mothernonallocatedatoms.jumps)[1],
  )

  state.mothernonallocatedatoms.locations =
    zeros(size(state.mothernonallocatedatoms.jumps)[1])
end

function updateauxu!(state::MCMCState, input::MCMCInput)
  # For each group, compute the sum of all the jumps of the corresponding
  # children process.
  sumjumps =
    map(x -> sum(x.jumps), state.childrenallocatedatoms) +
    map(x -> sum(x.jumps), state.childrennonallocatedatoms)

  #= Create g Gamma distributions, each one with shape equal to the number of
    observations in that group and rate equal to the sum of all the jumps of
    the children process of that group. We write the inverse of the second
    parameter because the Gamma() function requires the shape and the scale. =#
  gammas = Gamma.(input.n, 1 ./ sumjumps)

  state.auxu[:] = rand.(gammas, 1)
end

function hsncpmixturemodel_fit(
  input::MCMCInput,
  model::Model;
  iterations = iterations,
  burnin = burnin,
  thin = thin,
)
  state = MCMCState(input.g, input.n)
  initalizemcmcstate!(state, model)

  output = MCMCOutput(iterations, input.g, input.n, model)

  for it = 1:(burnin+iterations*thin)
    updatemotherprocess!(state, model)

    # TODO: update MCMC state

    updateauxu!(state, input)

    if it > burnin && mod(it, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
    end
  end

  return output
end
