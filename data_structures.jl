# This file contains all the data structures that are used in the sampler and
# their auxiliary functions.

# Structure that contains the input of the MCMC.
struct MCMCInput
  # A vector containing, for each group, the corresponding observations.
  data::Vector{Vector{Real}}
  # A vector containing the number of observations in each group.
  n::Vector{Integer}
  # The number of groups.
  g::Integer

  function MCMCInput(data)
    return new(data, [size(data[l])[1] for l = 1:size(data)[1]], size(data)[1])
  end
end

struct AtomsContainer
  jumps::Vector{Real}
  locations::Array{Real}
  # Count how many atoms (or observations) are linked to this atoms through the
  # clustering labels.
  counter::Vector{Integer}

  function AtomsContainer()
    new(Real[], Real[], Integer[])
  end
end

function deleteat!(atoms::AtomsContainer; index = h)
  jump = deleteat!(atoms.jumps, h)
  location = deleteat!(atoms.locations, h)
  counter = deleteat!(atoms.counter, h)
end

function push!(
  atoms::AtomsContainer;
  jump = jump,
  location = location,
  counter = counter,
)
  push!(atoms.jumps, jump)
  push!(atoms.locations, location)
  push!(atoms.counter, counter)
end

struct MCMCState
  # A vector containing, for each group, the auxiliary variable called u_l.
  auxu::Vector{Real}
  # A vector containing the within-group clustering labels for each observation.
  wgroupcluslabels::Vector{Vector{Integer}}
  # A vector containing, for each group, the clustering labels for each
  # allocated atom of the children process.
  childrenatomslabels::Vector{Vector{Integer}}
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
  motherallocatedatoms::AtomsContainer
  # A vector containing the non allocated atoms of the mother process.
  mothernonallocatedatoms::AtomsContainer

  function MCMCState(g, n)
    new(
      zeros(g),
      [zeros(n[l]) for l = 1:g],
      fill(Integer[], g),
      AtomsContainer[],
      AtomsContainer[],
      AtomsContainer(),
      AtomsContainer(),
    )
  end
end

function deallocatechildrenatom!(state::MCMCState; group = l, index = h)
  #= This function removes the h-indexed allocated atom of the children process
    of group l from the list of allocated atoms. It also modifies the other
    elements of the state to maintain coherency. =#

  # Remove all the information of the selected atom from the list of allocated
  # atoms.
  jump, location, _ = deleteat!(state.childrenallocatedatoms[l], h)

  # Add the selected atom to the list of non allocated atoms.
  push!(state.childrennonallocatedatoms[l], jump, location, 0)

  # Remove the clustering label for the selected atom. Indeed, the state does
  # not contain clustering labels for non allocated atoms.
  deleteat!(state.childrenatomslabels[l], h)

  #= Given that an allocated atom has been removed, the other allocated atoms
    with index greater than h had their index reduced by 1. Thus, we reduce the
    corresponding index in the vector of the within-group clustering labels. =#
  state.wgroupcluslabels[l] =
    state.wgroupcluslabels[l] - (state.wgroupcluslabels[l] .> h)
end

function allocatechildrenatom!(state::MCMCState; group = l, index = h)
  #= This function adds the h-indexed non allocated atom of the children process
    of group l to the list of allocated atoms. It also modifies the other
    elements of the state to maintain coherency.
    This function does not change the clustering labels, it just initializes the
    counter of the selected atom to 1. =#

  # Remove all the information of the selected atom from the list of non
  # allocated atoms.
  jump, location, _ = deleteat!(state.childrennonallocatedatoms[l], h)

  # Add the selected atom to the list of non allocated atoms.
  push!(state.childrenallocatedatoms[l], jump, location, 1)
end

function deallocatemotheratom!(state::MCMCState; index = j)
  #= This function removes the j-indexed allocated atom of the mother process
    from the list of allocated atoms. It also modifies the other
    elements of the state to maintain coherency. =#

  # Remove all the information of the selected atom from the list of allocated
  # atoms.
  jump, location, _ = deleteat!(state.motherallocatedatoms, j)

  # Add the selected atom to the list of non allocated atoms.
  push!(state.mothernonallocatedatoms, jump, location, 0)

  #= Given that an allocated atom has been removed, the other allocated atoms
    with index greater than j had their index reduced by 1. Thus, we reduce the
    corresponding index in the vector of the children atoms clustering labels.
    We use the map() function because this operation must be repeated for every
    group. =#
  state.childrenatomslabels = map(x -> x .- (x .> j), state.childrenatomslabels)
end

function allocatemotheratom!(state::MCMCState; index = j)
  #= This function adds the j-indexed non allocated atom of the mother process
    to the list of allocated atoms. It also modifies the other elements of the
    state to maintain coherency.
    This function does not change the clustering labels, it just initializes the
    counter of the selected atom to 1. =#

  # Remove all the information of the selected atom from the list of non
  # allocated atoms.
  jump, location, _ = deleteat!(state.mothernonallocatedatoms, j)

  # Add the selected atom to the list of non allocated atoms.
  push!(state.motherallocatedatoms, jump, location, 1)
end

struct MCMCOutput
  # A matrix containing, for each iteration, the mixture component parameter's
  # values for each observation.
  cluslocations::Array{Array{Real}}
  # A matrix containing, for each iteration, the within-group clustering label
  # for each observation.
  wgroupcluslabels::Array{Array{Integer}}
  # A matrix containing, for each iteration, the across-group clustering label
  # for each observation.
  agroupcluslabels::Array{Array{Integer}}

  function MCMCOutput(iterations, g, n, model::NormalMeanModel)
    new(
      [zeros(iterations, n[l], 1) for l = 1:g],
      [zeros(iterations, n[l]) for l = 1:g],
      [zeros(iterations, n[l]) for l = 1:g],
    )
  end
end

function updatemcmcoutput!(
  input::MCMCInput,
  state::MCMCState,
  output::MCMCOutput,
  idx::Integer,
)
  for l = 1:input.g
    # We use deepcopy because by default Julia copies only the reference to
    # data, but the state will change at each iteration.

    # Copy the within-group clustering labels from the state.
    output.wgroupcluslabels[l][idx, :] = deepcopy(state.wgroupcluslabels[l])
    # Obtain the within-group cluster locations for each observations using the
    # clustering labels.
    output.cluslocations[l][idx, :] = deepcopy(
      state.childrenallocatedatoms[l].locations[state.wgroupcluslabels[l]],
    )
    # Obtain the across-group cluster locations for each observations using the
    # clustering labels.
    output.agroupcluslabels[idx, rangeobs] =
      deepcopy(state.childrenatomslabels[l][state.wgroupcluslabels[l]])
  end
end
