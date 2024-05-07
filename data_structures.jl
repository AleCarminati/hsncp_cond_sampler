# This file contains all the data structures that are used in the sampler and
# their auxiliary functions.

# Structure that contains the input of the MCMC.
struct MCMCInput
  # A vector containing, for each group, the corresponding observations.
  data::Vector{Vector{Float64}}
  # A vector containing the number of observations in each group.
  n::Vector{Int32}
  # The number of groups.
  g::Int32

  function MCMCInput(data)
    return new(data, [size(data[l])[1] for l = 1:size(data)[1]], size(data)[1])
  end
end

mutable struct AtomsContainer
  jumps::Vector{Float64}
  locations::Vector{Vector{Float64}}
  # Count how many atoms (or observations) are linked to this atoms through the
  # clustering labels.
  counter::Vector{Int32}

  function AtomsContainer()
    new(Float64[], Float64[], Int32[])
  end
end

function deleteatatomcont!(atoms::AtomsContainer, h)
  jump = atoms.jumps[h]
  deleteat!(atoms.jumps, h)
  location = atoms.locations[h]
  deleteat!(atoms.locations, h)
  counter = atoms.counter[h]
  deleteat!(atoms.counter, h)
  return jump, location, counter
end

function pushatomcont!(atoms::AtomsContainer, jump, location, counter)
  push!(atoms.jumps, jump)
  push!(atoms.locations, location)
  push!(atoms.counter, counter)
end

mutable struct MCMCState
  # A vector containing all the parameters of the mixture function.
  mixtparams::Vector{Float64}
  # A vector containing, for each group, the auxiliary variable called u_l.
  const auxu::Vector{Float64}
  # A vector containing the within-group clustering labels for each observation.
  const wgroupcluslabels::Vector{Vector{Int32}}
  # A vector containing, for each group, the clustering labels for each
  # allocated atom of the child process.
  childrenatomslabels::Vector{Vector{Int32}}
  #= A vector containing, for each group, the allocated atoms of the children
  process. A atom is allocated if it is linked to at least one observation
  through the clustering labels. =#
  childrenallocatedatoms::Vector{AtomsContainer}
  # A matrix containing, for each group, the non allocated atoms of the children
  # process.
  childrennonallocatedatoms::Vector{AtomsContainer}
  #= A vector containing the allocated atoms of the mother process. A atom is
  allocated if it is linked to at least one allocated atom of the children
  processes throught the clustering labels. =#
  const motherallocatedatoms::AtomsContainer
  # A vector containing the non allocated atoms of the mother process.
  mothernonallocatedatoms::AtomsContainer

  function MCMCState(g, n)
    new(
      Float64[],
      zeros(g),
      [zeros(n[l]) for l = 1:g],
      [Int32[] for _ = 1:g],
      [AtomsContainer() for _ = 1:g],
      [AtomsContainer() for _ = 1:g],
      AtomsContainer(),
      AtomsContainer(),
    )
  end
end

function getalljumps(state::MCMCState; group = nothing)
  # Returns a vector containing all the jumps of the child process (if group is
  # specified) or the mother process (if group is not specified).
  if group == nothing
    return vcat(
      state.motherallocatedatoms.jumps,
      state.mothernonallocatedatoms.jumps,
    )
  else
    return vcat(
      state.childrenallocatedatoms[group].jumps,
      state.childrennonallocatedatoms[group].jumps,
    )
  end
end

function getalllocs(state::MCMCState; group = nothing)
  # Returns a vector containing all the locations of the child process (if
  # group is specified) or the mother process (if group is not specified).
  if group == nothing
    return vcat(
      state.motherallocatedatoms.locations,
      state.mothernonallocatedatoms.locations,
    )
  else
    return vcat(
      state.childrenallocatedatoms[group].locations,
      state.childrennonallocatedatoms[group].locations,
    )
  end
end

function deallocatechildrenatom!(state::MCMCState, l, h)
  #= This function removes the h-indexed allocated atom of the child process
    of group l from the list of allocated atoms. It also modifies the other
    elements of the state to maintain coherency. =#

  # Remove all the information of the selected atom from the list of allocated
  # atoms.
  jump, location, _ = deleteatatomcont!(state.childrenallocatedatoms[l], h)

  # Add the selected atom to the list of non allocated atoms.
  pushatomcont!(state.childrennonallocatedatoms[l], jump, location, 0)

  # Get the index of the corresponding mother process atom.
  idxmotheratom = state.childrenatomslabels[l][h]
  # Decrease the counter of the corresponding mother process atom.
  state.motherallocatedatoms.counter[idxmotheratom] -= 1
  # If the corresponding mother process atom does not have any associated
  # child processes' atoms, deallocate it.
  if state.motherallocatedatoms.counter[idxmotheratom] == 0
    deallocatemotheratom!(state, idxmotheratom)
  end

  # Remove the clustering label for the selected atom. Indeed, the state does
  # not contain clustering labels for non allocated atoms.
  deleteat!(state.childrenatomslabels[l], h)

  #= Given that an allocated atom has been removed, the other allocated atoms
    with index greater than h had their index reduced by 1. Thus, we reduce the
    corresponding index in the vector of the within-group clustering labels. =#
  state.wgroupcluslabels[l] =
    state.wgroupcluslabels[l] - (state.wgroupcluslabels[l] .> h)
end

function allocatechildrenatom!(state::MCMCState, l, h, atomlabel)
  #= This function adds the h-indexed non allocated atom of the child process
    of group l to the list of allocated atoms. It also modifies the other
    elements of the state to maintain coherency.
    This function does not change the clustering labels and it initializes the
    counter of the selected atom to 0. =#

  # Remove all the information of the selected atom from the list of non
  # allocated atoms.
  jump, location, _ = deleteatatomcont!(state.childrennonallocatedatoms[l], h)

  # Add the clustering label for the selected atom.
  push!(state.childrenatomslabels[l], atomlabel)

  # Increase the counter of the corresponding mother process atom.
  state.motherallocatedatoms.counter[atomlabel] += 1

  # Add the selected atom to the list of non allocated atoms.
  pushatomcont!(state.childrenallocatedatoms[l], jump, location, 1)
end

function deallocatemotheratom!(state::MCMCState, j)
  #= This function removes the j-indexed allocated atom of the mother process
    from the list of allocated atoms. It also modifies the other
    elements of the state to maintain coherency. =#

  # Remove all the information of the selected atom from the list of allocated
  # atoms.
  jump, location, _ = deleteatatomcont!(state.motherallocatedatoms, j)

  # Add the selected atom to the list of non allocated atoms.
  pushatomcont!(state.mothernonallocatedatoms, jump, location, 0)

  #= Given that an allocated atom has been removed, the other allocated atoms
    with index greater than j had their index reduced by 1. Thus, we reduce
    the corresponding index in the vector of the children atoms clustering
    labels. We use the map() function because this operation must be repeated
    for every group. =#
  state.childrenatomslabels =
    map(x -> @.(x - (x > j)), state.childrenatomslabels)
end

function allocatemotheratom!(state::MCMCState, j)
  #= This function adds the j-indexed non allocated atom of the mother process
    to the list of allocated atoms. It also modifies the other elements of the
    state to maintain coherency.
    This function does not change the clustering labels and it initializes the
    counter of the selected atom to 0. =#

  # Remove all the information of the selected atom from the list of non
  # allocated atoms.
  jump, location, _ = deleteatatomcont!(state.mothernonallocatedatoms, j)

  # Add the selected atom to the list of non allocated atoms.
  pushatomcont!(state.motherallocatedatoms, jump, location, 0)
end

struct MCMCOutput
  # A (iterations, n_mixtparams) matrix containing, for each iteration,
  # all the parameters of the mixture function.
  mixtparams::Matrix{Float64}
  # A g-length vector of (iterations, n_l, dim_childloc) arrays containing, for
  # each iteration, the mixture component parameter's values for each
  # observation.
  cluslocations::Vector{Array{Float64}}
  # A g-length vector of (iterations, n_l) containing, for each iteration, the
  # within-group clustering label for each observation.
  wgroupcluslabels::Vector{Matrix{Int32}}
  # A g-length vector of (iterations, n_l) containing, for each iteration, the
  # within-group clustering label for each observation.
  agroupcluslabels::Array{Array{Int32}}
  # A vector containing, for each iteration, the allocated atoms of the mother
  # process.
  motherallocatedatoms::Vector{AtomsContainer}

  function MCMCOutput(iterations, g, n, model::GammaCRMModel)
    new(
      zeros(iterations, 1),
      [zeros(iterations, n[l], 1) for l = 1:g],
      [zeros(iterations, n[l]) for l = 1:g],
      [zeros(iterations, n[l]) for l = 1:g],
      [AtomsContainer() for it = 1:iterations],
    )
  end
end

function updatemcmcoutput!(
  input::MCMCInput,
  state::MCMCState,
  output::MCMCOutput,
  idx::Integer,
)
  # We use deepcopy because by default Julia copies only the reference to
  # data, but the state will change at each iteration.

  # Save the mixture parameters only when there are some non-fixed mixture
  # parameters.
  if size(state.mixtparams)[1] > 0
    output.mixtparams[idx, :] = deepcopy(state.mixtparams)
  end

  for l = 1:input.g
    # Copy the within-group clustering labels from the state.
    output.wgroupcluslabels[l][idx, :] = deepcopy(state.wgroupcluslabels[l])

    # Obtain the within-group cluster locations for each observations using the
    # clustering labels. We have to use stack() to transform the vector of
    # vectors (containing the locations) in a matrix.
    output.cluslocations[l][idx, :, :] = deepcopy(
      stack(
        state.childrenallocatedatoms[l].locations[state.wgroupcluslabels[l]],
        dims = 1,
      ),
    )

    # Obtain the across-group cluster locations for each observations using the
    # clustering labels.
    output.agroupcluslabels[l][idx, :] =
      deepcopy(state.childrenatomslabels[l][state.wgroupcluslabels[l]])

    output.motherallocatedatoms[idx] = deepcopy(state.motherallocatedatoms)
  end
end
