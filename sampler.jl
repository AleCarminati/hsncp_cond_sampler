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
    standevs .^ 2

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

  state.mothernonallocatedatoms.counter =
    zeros(size(state.mothernonallocatedatoms.jumps)[1])
end

function updatechildrenprocesses!(
  input::MCMCInput,
  state::MCMCState,
  model::Model,
)
  g = size(state.childrenallocatedatoms)[1]
  for l = 1:g
    # Update the allocated atoms of the mother process.
    updatechildprocessalloc!(input, state, model, l)
    # Update the non allocated atoms of the mother process.
    updatechildprocessnonalloc!(state, model, l)
  end
end

function updatechildprocessalloc!(
  input::MCMCInput,
  state::MCMCState,
  model::NormalMeanModel,
  l,
)
  g = size(state.auxu)[1]
  nallocatoms = size(state.childrenallocatedatoms[l].jumps)[1]

  # First, sample the jumps.

  shapes = model.childenjumpshape .+ state.childrenallocatedatoms[l].counter
  scales = fill(1 / (model.childenjumprate + state.auxu[l]))

  gammas = Gamma.(shapes, scales)

  state.childrenallocatedatoms[l].jumps[:] = rand.(gammas, 1)

  # Then, sample the locations.

  standevs =
    sqrt.(
      1.0 / (
        1 / model.kernelsd^2 .+
        state.childrenallocatedatoms[l].counter ./ model.mixturecompsd^2
      )
    )

  # For each allocated atom of the child process, compute the sum of the
  # observations associated with it.
  sums = map(
    x -> transpose(input.data[l]) * (state.wgroupcluslabels[l] .== x),
    Vector(1:nallocatoms),
  )

  means =
    standevs .^ 2 .* (
      state.motherallocatedatoms.locations[state.childrenatomslabels[l]] ./
      model.kernelsd^2 .+ sums ./ model.mixturecompsd^2
    )

  normals = Normal.(means, standevs)

  state.childrenallocatedatoms[l].locations[:] = rand.(normals, 1)
end

function updatechildprocessnonalloc!(
  state::MCMCState,
  model::NormalMeanModel,
  l,
)
  new_rate = model.childenjumprate + state.auxu[l]

  coef = (model.childenjumprate / new_rate)^model.childrenjumpshape

  f =
    x ->
      coef * gamma(model.childrenjumpshape, new_rate * x) /
      gamma(model.childrenjumpshape)

  state.childrennonallocatedatoms[l].jumps = fergusonklass(f, 0.1)

  # For each non allocated atom of the children process, sample which atom of
  # the mother process it is associated with, using the jumps of the mother
  # process as weights.
  motheratomsjumps =
    hcat(state.motherallocatedatoms.jumps, state.mothernonallocatedatoms.jumps)
  weights = motheratomsjumps ./ sum(motheratomsjumps)
  associations = rand(
    Categorical(weights),
    size(state.childrennonallocatedatoms[l].jumps)[1],
  )
  # Then, set up the distribution of the children process atoms based on the
  # Gaussian kernels centered in the associated atom of the mother process.
  motheratomslocations = hcat(
    state.motherallocatedatoms.locations,
    state.mothernonallocatedatoms.locations,
  )
  normals = Normal(
    motheratomslocations[associations],
    fill(model.kernelsd, size(state.childrennonallocatedatoms[l].jumps)[1]),
  )
  # Eventually, sample the locations, using the distributions created in the
  # last lines.
  state.childrennonallocatedatoms.locations = rand.(normals, 1)

  state.childrennonallocatedatoms.counter =
    zeros(size(state.childrennonallocatedatoms[l].jumps)[1])
end

function updatechildrenatomslabels!(state::MCMCState, model::NormalMeanModel)
  g = size(state.auxu)[1]

  for l = 1:g
    jumps = hcat(
      state.motherallocatedatoms.jumps,
      state.mothernonallocatedatoms.jumps,
    )
    locations = hcat(
      state.motherallocatedatoms.locations,
      state.mothernonallocatedatoms.locations,
    )
    normals = Normal.(locations, fill(model.kernelsd, size(locations)[1]))
    nalloc = size(state.motherallocatedatoms.jumps)[1]
    for i = 1:size(state.childrenallocatedatoms[l].location)[1]
      probs =
        pdf.(normals, state.childrenallocatedatoms[l].location[i]) .* jumps
      probs = probs ./ sum(probs)

      sampledidx = rand(Categorical(probs))

      # Save the old clustering label of the atom.
      oldidx = state.childrenatomslabels[l][i]

      if sampledidx <= nalloc
        # The sampled atom is already an allocated atom.
        state.childrenatomslabels[l][i] = sampledidx
        state.motherallocatedatoms[sampledidx].counter += 1
      else
        # The sampled atom is a new atom.
        allocatemotheratom!(state; index = sampledidx - nalloc)
        nalloc += 1
        state.childrenatomslabels[l][i] = nalloc
      end

      state.motherallocatedatoms[oldidx].counter -= 1
      # If the children atom was the only one associated with a certain atom
      # of the mother process, remove the latter from the list of allocated
      # atoms.
      if state.motherallocatedatoms[oldidx].counter == 0
        deallocatemotheratom!(state; index = oldidx)
        nalloc -= 1
      end
    end
  end
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

    updatechildrenprocesses!(input, state, model)

    updatechildrenatomslabels!(state, model)

    # TODO: update MCMC state

    updateauxu!(state, input)

    if it > burnin && mod(it, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
    end
  end

  return output
end
