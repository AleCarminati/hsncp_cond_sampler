using Random

# Structure that contains the input of the MCMC.
struct MCMCInput
  # A vector containing all the observations.
  data::Array{Real}
  # A vector containing the number of observations in each group.
  n::Array{Integer}
  # The number of groups.
  g::Integer

  function MCMCInput(data, n, g)
    if (size(n)[1] != g)
      throw(
        DomainError(
          "The number of groups and the size of the array " *
          "containing the number of observations in each group are not coherent.",
        ),
      )
    end
    return new(data, n, g)
  end
end

struct MCMCOutput
  # A matrix containing, for each iteration, the mixture component parameter's
  # values for each observation.
  cluslocations::Array{Real}
  # A matrix containing, for each iteration, the within-group clustering label
  # for each observation.
  wgroupclusalloc::Array{Integer}
  # A matrix containing, for each iteration, the across-group clustering label
  # for each observation.
  agroupclusalloc::Array{Integer}
  function MCMCOutput(iterations, numdata, dimchildrenloc)
    new(
      zeros(iterations, numdata, dimchildrenloc),
      zeros(iterations, numdata),
      zeros(iterations, numdata),
    )
  end
end

struct AtomsContainer
  jumps::Array{Real}
  locations::Array{Real}
  # Count how many atoms (or observations) are linked to this atoms through the
  # clustering labels.
  counter::Array{Integer}
end

struct MCMCState
  # A vector containing the auxiliary variables called u_l, one for each group.
  auxu::Array{Real}
  # A vector containing the within-group clustering labels for each observation.
  wgroupclusalloc::Array{Integer}
  # A matrix containing, for each group, the clustering labels for each
  # allocated atom of the children process.
  childrenatomsallocation::Array{Integer}
  #= A matrix containing, for each group, the allocated atoms of the children
  		process. A atom is allocated if it is linked to at least one observation
  		through the clustering labels. =#
  childrenallocatedatoms::Array{AtomsContainer}
  # A matrix containing, for each group, the non allocated atoms of the children
  # process.
  childrennonallocatedatoms::Array{AtomsContainer}
  #= A vector containing the allocated atoms of the mother process. A atom is
  		allocated if it is linked to at least one allocated atom of the children
  		processes throught the clustering labels. =#
  motherallocatedatoms::Array{AtomsContainer}
  # A vector containing the non allocated atoms of the mother process.
  mothernonallocatedatoms::Array{AtomsContainer}

  function MCMCState(g, numdata)
    new(
      zeros(g),
      zeros(numdata),
      Integer[],
      AtomsContainer[],
      AtomsContainer[],
      AtomsContainer[],
      AtomsContainer[],
    )
  end
end

function initalizemcmcstate!(state::MCMCState)
  # TODO: initalization of the state of the MCMC.
end

function hsncpmixturemodel_fit(
  input::MCMCInput;
  iterations = iterations,
  dimchildrenloc = dimchildrenloc,
)
  state = MCMCState(input.g, sum(input.n))
  output = MCMCOutput(iterations, sum(input.n), dimchildrenloc)

  initalizemcmcstate!(state)

  # TODO: run the MCMC.

  return output
end
