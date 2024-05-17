# This file contains all the data structures that are used in the sampler and
# their auxiliary functions.

function moveinplace!(vec, startidx, endidx)
  # Moves the element at index startidx of vector vec to index endidx.
  # Performs this move efficiently, avoiding allocations.

  signmove = sign(endidx - startidx)

  if signmove != 0
    temp = vec[startidx]

    for idx = startidx:signmove:(endidx-signmove)
      vec[idx] = vec[idx+signmove]
    end

    vec[endidx] = temp
  end

  return nothing
end

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

function deleteandpushatomcont!(
  startatoms::AtomsContainer,
  endatoms::AtomsContainer,
  idx,
)
  # Delete the atom at index idx in startatoms and push it in endatoms, with a
  # 0 counter.

  jump = startatoms.jumps[idx]
  deleteat!(startatoms.jumps, idx)
  location = startatoms.locations[idx]
  deleteat!(startatoms.locations, idx)
  deleteat!(startatoms.counter, idx)

  push!(endatoms.jumps, jump)
  push!(endatoms.locations, location)
  push!(endatoms.counter, 0)
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
  # A vector containing, for each group, its associated mother process label.
  const groupcluslabels::Vector{Int32}
  # A vector containing the probability for a generic group to be assigned to
  # a mother atom.
  const probsgroupclus::Vector{Float64}
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
  const motherallocatedatoms::Vector{AtomsContainer}
  # A vector containing the non allocated atoms of the mother process.
  mothernonallocatedatoms::Vector{AtomsContainer}

  function MCMCState(g, n, m)
    new(
      Float64[],
      zeros(g),
      [zeros(n[l]) for l = 1:g],
      [Int32[] for _ = 1:g],
      zeros(g),
      zeros(m),
      [AtomsContainer() for _ = 1:g],
      [AtomsContainer() for _ = 1:g],
      [AtomsContainer() for _ = 1:m],
      [AtomsContainer() for _ = 1:m],
    )
  end
end

function getatomscont(
  state::MCMCState;
  group = nothing,
  motherprocess = nothing,
)
  # Returns the AtomsContainer of the allocated atoms and of the non allocated
  # atoms of the child process (if group is specified) or the mother process
  # (if group is not specified) specified by motherprocess.
  if group == nothing
    allocatedatoms = state.motherallocatedatoms[motherprocess]
    nonallocatedatoms = state.mothernonallocatedatoms[motherprocess]
  else
    allocatedatoms = state.childrenallocatedatoms[group]
    nonallocatedatoms = state.childrennonallocatedatoms[group]
  end

  return allocatedatoms, nonallocatedatoms
end

function getalljumps(state::MCMCState; group = nothing, motherprocess = nothing)
  # Returns a vector containing all the jumps of the child process (if group is
  # specified) or the mother process (if group is not specified) specified by
  # motherprocess.

  allocatedatoms, nonallocatedatoms =
    getatomscont(state, group = group, motherprocess = motherprocess)

  return vcat(allocatedatoms.jumps, nonallocatedatoms.jumps)
end

function getalllocs(state::MCMCState; group = nothing, motherprocess = nothing)
  # Returns a vector containing all the locations of the child process (if
  # group is specified) or the mother process (if group is not specified).

  allocatedatoms, nonallocatedatoms =
    getatomscont(state, group = group, motherprocess = motherprocess)

  return vcat(allocatedatoms.locations, nonallocatedatoms.locations)
end

function deallocateatom!(
  state::MCMCState,
  idx;
  group = nothing,
  motherprocess = nothing,
)
  #= Removes the allocated atom with index idx of the child process (if group
    is specified) or the mother process (if group is not specified) from the
    list of the allocated atoms. It also modifies the other elements of the
    state to maintain coherency. =#

  allocatedatoms, nonallocatedatoms =
    getatomscont(state, group = group, motherprocess = motherprocess)

  # Move the atom from the list of allocated atoms to the list of non allocated
  # ones.
  deleteandpushatomcont!(allocatedatoms, nonallocatedatoms, idx)

  #= Given that an allocated atom has been removed, the other allocated atoms
    with index greater than idx had their index reduced by 1. Thus, we reduce
    the corresponding index in the vector of the children atoms clustering
    labels.=#
  if group == nothing
    for l in findall(state.groupcluslabels .== motherprocess)
      state.childrenatomslabels[l] .-= (state.childrenatomslabels[l] .> idx)
    end
  else
    # Maintain coherence with labels and counters of the mother process.
    # Get the index of the corresponding mother process atom.
    idxmotheratom = state.childrenatomslabels[group][idx]
    # Decrease the counter of the corresponding mother process atom.
    state.motherallocatedatoms[state.groupcluslabels[group]].counter[idxmotheratom] -=
      1
    # If the corresponding mother process atom does not have any associated
    # child processes' atoms, deallocate it.
    if state.motherallocatedatoms[state.groupcluslabels[group]].counter[idxmotheratom] ==
       0
      deallocateatom!(
        state,
        idxmotheratom,
        group = nothing,
        motherprocess = state.groupcluslabels[group],
      )
    end

    # Remove the clustering label for the selected atom. Indeed, the state does
    # not contain clustering labels for non allocated atoms.
    deleteat!(state.childrenatomslabels[group], idx)

    state.wgroupcluslabels[group] .-= (state.wgroupcluslabels[group] .> idx)
  end
end

function allocateatom!(
  state::MCMCState,
  idx;
  group = nothing,
  motherprocess = nothing,
)
  #= Adds the idx-indexed non allocated atom of the child process (if group is
    specified) or the mother process (if group is not specified) to the list of
    allocated atoms.
    This function does not change any clustering labels and it initializes the
    counter of the selected atom to 0. =#

  allocatedatoms, nonallocatedatoms =
    getatomscont(state, group = group, motherprocess = motherprocess)

  # Move the atom from the list of allocated atoms to the list of non allocated
  # ones.
  deleteandpushatomcont!(nonallocatedatoms, allocatedatoms, idx)
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
  # A (iterations, g) matrix containing, for each iterations, the cluster labels
  # for each group.
  groupcluslabels::Matrix{Int32}
  # A vector containing, for each iteration, the allocated atoms of the mother
  # process.
  motherallocatedatoms::Vector{Vector{AtomsContainer}}

  function MCMCOutput(iterations, g, n, model::GammaCRMModel)
    new(
      zeros(iterations, 1),
      [zeros(iterations, n[l], 1) for l = 1:g],
      [zeros(iterations, n[l]) for l = 1:g],
      [zeros(iterations, n[l]) for l = 1:g],
      zeros(iterations, g),
      [
        [AtomsContainer() for _ = 1:model.nmotherprocesses] for _ = 1:iterations
      ],
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

    output.groupcluslabels[idx, :] = deepcopy(state.groupcluslabels)

    for m in eachindex(state.motherallocatedatoms)
      output.motherallocatedatoms[idx][m] =
        deepcopy(state.motherallocatedatoms[m])
    end
  end
end
