using Random

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

function initalizemcmcstate!(state::MCMCState)
  # TODO: initalization of the state of the MCMC.
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
  function MCMCOutput(iterations, g, n, dimchildrenloc)
    new(
      [zeros(iterations, n[l], dimchildrenloc) for l = 1:g],
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

function hsncpmixturemodel_fit(
  input::MCMCInput;
  dimchildrenloc = dimchildrenloc,
  iterations = iterations,
  burnin = burnin,
  thin = thin,
)
  state = MCMCState(input.g, input.n)
  initalizemcmcstate!(state)

  output = MCMCOutput(iterations, input.g, input.n, dimchildrenloc)

  for it = 1:(burnin+iterations*thin)
    # TODO: update MCMC state

    if it > burnin && mod(it, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
    end
  end

  return output
end
