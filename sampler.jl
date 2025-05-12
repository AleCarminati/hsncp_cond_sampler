# This file contains the functions to run the conditional sampler for the
# HSNCP mixture model.

function getatomcenterednormals(
  state::MCMCState,
  model::Model;
  group = nothing,
  motherprocess = nothing,
)
  #= This function returns a vector of Normal distributions.
    - group not specified: it returns Normal kernels centered in the means in
      the mother process' atoms, with variance equal to the variance in the
      mother process' atoms.
    - group specified: it returns Normal mixture components centered in the
      locations of the specified group child process' atoms. =#

  means = getprocessmeans(
    state,
    model,
    group = group,
    motherprocess = motherprocess,
    onlyalloc = false,
  )
  vars = getprocessvars(
    state,
    model,
    group = group,
    motherprocess = motherprocess,
    onlyalloc = false,
  )

  return Normal.(means, sqrt.(vars))
end

function getprocessvars end
#= Returns a vector filled with the variance of the kernel (group = nothing) or
  the variance of the mixture Gaussian (group = l).
  The length of the vector matches the number of atoms considered, based
  on the combination of group and onlyalloc.
  With certain models, the variances are fixed: in that case it returns a
  vector containing the fixed value with length equal to the number of
  considered atoms. =#

function getprocessvars(
  state::MCMCState,
  model::NormalMeanModel;
  group = nothing,
  motherprocess = nothing,
  onlyalloc = true,
)
  if group == nothing
    var = model.kernelsd^2
    length = size(state.motherallocatedatoms[motherprocess].jumps)[1]
    if !onlyalloc
      length += size(state.mothernonallocatedatoms[motherprocess].jumps)[1]
    end
  else
    var = model.mixturecompsd^2
    length = size(state.childrenallocatedatoms[group].jumps)[1]
    if !onlyalloc
      length += size(state.childrennonallocatedatoms[group].jumps)[1]
    end
  end

  return fill(var, length)
end

function getprocessvars(
  state::MCMCState,
  model::Union{NormalMeanVarModel,NormalMeanVarVarModel};
  group = nothing,
  motherprocess = nothing,
  onlyalloc = true,
)
  if group == nothing
    vars = map(x -> x[2], state.motherallocatedatoms[motherprocess].locations)
    if !onlyalloc
      vars = vcat(
        vars,
        map(x -> x[2], state.mothernonallocatedatoms[motherprocess].locations),
      )
    end
  else
    length = size(state.childrenallocatedatoms[group].jumps)[1]
    if !onlyalloc
      length += size(state.childrennonallocatedatoms[group].jumps)[1]
    end
    vars = fill(getmixtvar(state, model), length)
  end
  return vars
end

function getprocessmeans(
  state::MCMCState,
  model::Model;
  group = nothing,
  motherprocess = nothing,
  onlyalloc = true,
)
  #= Function that returns a vector containing the means that are saved in the
    atoms of the mother (group = nothing) or the child (group = l) process.
    If onlyalloc = true, it returns a vector containing the means that are
    saved only in the allocated atoms. =#
  if group == nothing
    means = map(x -> x[1], state.motherallocatedatoms[motherprocess].locations)
    if !onlyalloc
      means = vcat(
        means,
        map(x -> x[1], state.mothernonallocatedatoms[motherprocess].locations),
      )
    end
  else
    means = map(x -> x[1], state.childrenallocatedatoms[group].locations)
    if !onlyalloc
      means = vcat(
        means,
        map(x -> x[1], state.childrennonallocatedatoms[group].locations),
      )
    end
  end
  return means
end

function getmixtvar end
# Returns the value of the variance of the mixture component.

function getmixtvar(
  state::MCMCState,
  model::Union{NormalMeanModel,NormalMeanVarModel},
)
  return model.mixturecompsd^2
end

function getmixtvar(state::MCMCState, model::NormalMeanVarVarModel)
  return state.mixtparams[1]
end

function initializemcmcstate!(input::MCMCInput, state::MCMCState, model::Model)
  state.mixtparams = samplepriormixtparams(model)

  # The number of atoms to sample for each mother process and for each child
  # process.
  mothernatoms = 10
  childnatoms = 5

  # Sample the atoms of the mother process.
  motheratomsjumps = [
    rand(Distributions.Gamma(1, 1), mothernatoms) for
    _ = 1:model.nmotherprocesses
  ]
  motheratomslocations =
    [samplepriormotherloc(model, mothernatoms) for _ = 1:model.nmotherprocesses]

  # Contains the frequency of the sampled labels for the mother process.
  motherfreq = [zeros(mothernatoms) for _ = 1:model.nmotherprocesses]

  probsgroupclus = samplepriorprobgroupclus(model)
  groupcluslabels = rand(Categorical(probsgroupclus), input.g)

  for l = 1:input.g
    #= The sampled labels go from 1 to mothernatoms, while the labels of the
      state should go from 1 to the number of allocated atoms of the mother
      process. =#
    tempmotherlabels =
      rand(Distributions.DiscreteUniform(1, mothernatoms), childnatoms)

    childatomsjumps = rand(Distributions.Gamma(1, 1), childnatoms)

    # For each atom of the child process, sample its location based on
    # the associated atom of the mother process.
    childatomslocations = samplechildloc(
      model,
      motheratomslocations[groupcluslabels[l]][tempmotherlabels],
    )

    #= The sampled labels go from 1 to childnatoms, while the labels of the
      state should go from 1 to the number of allocated atoms of the child
      process. =#
    tempchildlabels =
      rand(Distributions.DiscreteUniform(1, childnatoms), input.n[l])

    # Compute the frequency of the sampled labels.
    childfreq = StatsBase.counts(tempchildlabels, 1:childnatoms)

    # Extract the indexes of the atoms that have not been allocated.
    nonallocatomsidx = (1:childnatoms)[childfreq .== 0]
    # Extract the indexes of the atoms that have been allocated.
    allocatomsidx = (1:childnatoms)[childfreq .!= 0]
    #= Function that transforms the sampled labels (from 1 to childnatoms) in
      the correct within-group cluster labels (from 1 to the number of allocated
      atoms of the child process). =#
    obtaincluslabel = x -> findall(allocatomsidx .== x)[1]
    state.wgroupcluslabels[l] = deepcopy(map(obtaincluslabel, tempchildlabels))
    # Select the jumps and the locations of the allocated child process' atoms.
    state.childrenallocatedatoms[l].jumps =
      deepcopy(childatomsjumps[allocatomsidx])
    state.childrenallocatedatoms[l].locations =
      deepcopy(childatomslocations[allocatomsidx])
    # The counters will be the frequencies of the sampled labels for the
    # allocated atoms.
    state.childrenallocatedatoms[l].counter = deepcopy(childfreq[allocatomsidx])

    # Select the jumps and the locations of the non allocated child process'
    # atoms.
    state.childrennonallocatedatoms[l].jumps =
      deepcopy(childatomsjumps[nonallocatomsidx])
    state.childrennonallocatedatoms[l].locations =
      deepcopy(childatomslocations[nonallocatomsidx])
    state.childrennonallocatedatoms[l].counter =
      zeros(size(nonallocatomsidx)[1])

    # Update the frequency of the sampled labels for the mother process.
    motherfreq[groupcluslabels[l]] +=
      StatsBase.counts(tempmotherlabels[allocatomsidx], 1:mothernatoms)
    #= Save the sampled labels in the state, only for the allocated child
      process' atoms. Attention: at this stage, the sampled labels go from 1 to
      mothernatoms, therefore they are not valid. They will be correctly
      transformed between 1 and the number of allocated mother process' atoms at
      the end of the for loop, when it is know how many atoms are of the mother
      process are allocated. =#
    state.childrenatomslabels[l] = deepcopy(tempmotherlabels[allocatomsidx])

    state.auxu[l] = rand(Distributions.Gamma(input.n[l], sum(childatomsjumps)))
  end

  for m = 1:model.nmotherprocesses
    # Extract the indexes of the mother process' atoms that have not been
    # allocated.
    mothernonallocatomsidx = (1:mothernatoms)[motherfreq[m] .== 0]
    # Extract the indexes of the mother process' atoms that have been allocated.
    motherallocatomsidx = (1:mothernatoms)[motherfreq[m] .!= 0]

    #= Function that transforms the sampled labels (from 1 to mothernatoms) in
      the correct cluster labels for the child processes' atoms (from 1 to the
      number of allocated atoms of the mother process).
      Then, for each group, apply the function to the sampled labels vector. =#
    obtaincluslabel = x -> findfirst(motherallocatomsidx .== x)

    for l in findall(groupcluslabels .== m)
      state.childrenatomslabels[l] =
        map(obtaincluslabel, state.childrenatomslabels[l])
    end

    # Select the jumps and the locations of the allocated mother process' atoms.
    state.motherallocatedatoms[m].jumps =
      motheratomsjumps[m][motherallocatomsidx]
    state.motherallocatedatoms[m].locations =
      motheratomslocations[m][motherallocatomsidx]
    # The counters will be the frequencies of the sampled labels for the
    # allocated atoms.
    state.motherallocatedatoms[m].counter = motherfreq[m][motherallocatomsidx]

    # Select the jumps and the locations of the non allocated mother process'
    # atoms.
    state.mothernonallocatedatoms[m].jumps =
      motheratomsjumps[m][mothernonallocatomsidx]
    state.mothernonallocatedatoms[m].locations =
      motheratomslocations[m][mothernonallocatomsidx]
    state.mothernonallocatedatoms[m].counter =
      zeros(size(mothernonallocatomsidx)[1])
  end

  state.probsgroupclus .= probsgroupclus
  state.groupcluslabels .= groupcluslabels
end

function updatemixtparams!(
  input::MCMCInput,
  state::MCMCState,
  model::Union{NormalMeanModel,NormalMeanVarModel},
)
  return nothing
end

function updatemixtparams!(
  input::MCMCInput,
  state::MCMCState,
  model::NormalMeanVarVarModel,
)
  postshape = model.mixtshape + sum(input.n) / 2
  postscale =
    model.mixtscale +
    1 / 2 * sum(
      map(
        l -> sum(
          @. (
            input.data[l] -
            $getprocessmeans(state, model, group = l, onlyalloc = true)[state.wgroupcluslabels[l]]
          )^2
        ),
        Vector(1:input.g),
      ),
    )
  state.mixtparams[1] = rand(InverseGamma(postshape, postscale))
end

function updatemotherprocesses!(state::MCMCState, model::Model, fkthreshold)
  # Update the allocated atoms of the mother processes.
  updatemotherprocessesalloc!(state, model)
  # Update the non allocated atoms of the mother processes.
  updatemotherprocessesnonalloc!(state, model, fkthreshold)
end

function updatemotherprocessesalloc!(state::MCMCState, model::NormalMeanModel)
  # First, sample the jumps.
  updatemotherprocessesallocjumps!(state, model)

  # Then, sample the locations.
  updatemotherprocessesallocmeans!(state, model)
end

function updatemotherprocessesalloc!(
  state::MCMCState,
  model::Union{NormalMeanVarModel,NormalMeanVarVarModel},
)
  g = size(state.wgroupcluslabels)[1]

  # First, sample the jumps.
  updatemotherprocessesallocjumps!(state, model, model.motherprocess)

  # Then, sample the locations.
  updatemotherprocessesallocmeans!(state, model)
  updatemotherprocessesallocvars!(state, model)
end

function updatemotherprocessesallocjumps!(
  state::MCMCState,
  model::Model,
  process::GammaProcess,
)
  # Samples the jumps for the allocated atoms of the mother processes.
  for m in findall(
    StatsBase.counts(state.groupcluslabels, 1:model.nmotherprocesses) .> 0,
  )
    shape = state.motherallocatedatoms[m].counter
    rate =
      1 + sum(
        laplaceexp.(
          [model.childrenprocess],
          state.auxu[state.groupcluslabels .== m],
        ),
      )

    # We write the inverse of the second parameter because the Gamma() function
    # requires the shape and the scale.
    gammas = Gamma.(shape, 1 / rate)

    state.motherallocatedatoms[m].jumps[:] = rand.(gammas)
  end
end

function updatemotherprocessesallocjumps!(
  state::MCMCState,
  model::Model,
  motherprocess::GeneralizedGammaProcess,
)
  # Samples the jumps for the allocated atoms of the mother processes.
  for m in findall(
    StatsBase.counts(state.groupcluslabels, 1:model.nmotherprocesses) .> 0,
  )
    shape = state.motherallocatedatoms[m].counter .- motherprocess.sigma
    rate =
      motherprocess.tau .+ laplaceexp.(
        [model.childrenprocess],
        state.auxu[state.groupcluslabels .== m],
      )

    # We write the inverse of the second parameter because the Gamma() function
    # requires the shape and the scale.
    gammas = Gamma.(shape, 1 / rate)

    state.motherallocatedatoms[m].jumps[:] = rand.(gammas)
  end
end

function updatemotherprocessesallocmeans!(state::MCMCState, model::Model)
  for m in findall(
    StatsBase.counts(state.groupcluslabels, 1:model.nmotherprocesses) .> 0,
  )
    g = size(state.childrenallocatedatoms)[1]

    standevs = @. sqrt(
      1 / (
        1 / model.motherlocsd^2 +
        state.motherallocatedatoms[m].counter / $getprocessvars(
          state,
          model,
          group = nothing,
          motherprocess = m,
          onlyalloc = true,
        )
      ),
    )

    # For each allocated atom j of the mother process, compute the sum of all
    # the atoms of the child processes that are associated with j.
    sums = map(
      x -> sum(
        map(
          y ->
            transpose(state.childrenatomslabels[y] .== x) *
            getprocessmeans(state, model, group = y, onlyalloc = true),
          findall(state.groupcluslabels .== m),
        ),
      ),
      1:size(state.motherallocatedatoms[m].jumps)[1],
    )

    means = @. (
      model.motherlocmean / (model.motherlocsd^2) +
      sums / $getprocessvars(
        state,
        model,
        group = nothing,
        motherprocess = m,
        onlyalloc = true,
      )
    ) * standevs^2

    normals = Normal.(means, standevs)

    for i = 1:size(state.motherallocatedatoms[m].locations)[1]
      state.motherallocatedatoms[m].locations[i][1] = rand(normals[i])
    end
  end
end

function updatemotherprocessesallocvars!(
  state::MCMCState,
  model::Union{NormalMeanVarModel,NormalMeanVarVarModel},
)
  for m in findall(
    StatsBase.counts(state.groupcluslabels, 1:model.nmotherprocesses) .> 0,
  )
    shapes = @. model.motherlocshape + state.motherallocatedatoms[m].counter / 2

    # For each allocated atom j of the mother process, compute the sum of the
    # squared differences between atoms of the child processes that are associated
    # with j and the mean value for the atom j.
    sums = map(
      x -> sum(
        map(
          y -> sum(
            @.(
              (
                $getprocessmeans(state, model, group = y, onlyalloc = true)[state.childrenatomslabels[y]==x] -
                state.motherallocatedatoms[m].locations[x][1]
              )^2
            )
          ),
          findall(state.groupcluslabels .== m),
        ),
      ),
      1:size(state.motherallocatedatoms[m].jumps)[1],
    )

    scales = @. model.motherlocscale + sums / 2

    igammas = InverseGamma.(shapes, scales)

    for i = 1:size(state.motherallocatedatoms[m].locations)[1]
      state.motherallocatedatoms[m].locations[i][2] = rand(igammas[i])
    end
  end
end

function updatemotherprocessesnonalloc!(
  state::MCMCState,
  model::Model,
  fkthreshold,
)
  for m = 1:model.nmotherprocesses
    f =
      x -> motherprocessfkfun(
        x,
        m,
        model.motherprocess,
        model.childrenprocess,
        state,
      )

    state.mothernonallocatedatoms[m].jumps = fergusonklass(f, fkthreshold)

    state.mothernonallocatedatoms[m].locations = samplepriormotherloc(
      model,
      size(state.mothernonallocatedatoms[m].jumps)[1],
    )

    state.mothernonallocatedatoms[m].counter =
      zeros(size(state.mothernonallocatedatoms[m].jumps)[1])
  end
end

function motherprocessfkfun(
  x,
  m,
  motherprocess::GammaProcess,
  childprocess::Process,
  state,
)
  return motherprocess.totalmass * gamma(
    0,
    x * (
      1 +
      sum(@. laplaceexp([childprocess], state.auxu[state.groupcluslabels==m]))
    ),
  )
end

function motherprocessfkfun(
  x,
  motherprocess::GeneralizedGammaProcess,
  childprocess::Process,
  state,
)
  return motherprocess.totalmass / gamma(1 - motherprocess.sigma) * expint(
    1 + motherprocess.sigma,
    x * (
      motherprocess.tau +
      sum(@. laplaceexp([childprocess], state.auxu[state.groupcluslabels==m]))
    ),
  ) / x^motherprocess.sigma
end

function updatechildprocesses!(
  input::MCMCInput,
  state::MCMCState,
  model::Model,
  fkthreshold,
)
  for l = 1:input.g
    # Update the allocated atoms of the mother process.
    updatechildprocessalloc!(input, state, model, l)
    # Update the non allocated atoms of the mother process.
    updatechildprocessnonalloc!(state, model, l, fkthreshold)
  end
end

function updatechildprocessalloc!(
  input::MCMCInput,
  state::MCMCState,
  model::Model,
  l,
)
  updatechildprocessesallocjumps!(state, model.childrenprocess, l)

  # Then, sample the locations.
  motheratommean = getprocessmeans(
    state,
    model,
    group = nothing,
    motherprocess = state.groupcluslabels[l],
    onlyalloc = true,
  )
  motheratomvar = getprocessvars(
    state,
    model,
    group = nothing,
    motherprocess = state.groupcluslabels[l],
    onlyalloc = true,
  )
  childrenatomvar = getprocessvars(state, model, group = l, onlyalloc = true)

  standevs = @. sqrt(
    1.0 / (
      1 / motheratomvar[state.childrenatomslabels[l]] +
      state.childrenallocatedatoms[l].counter / childrenatomvar
    ),
  )

  nallocatoms = size(state.childrenallocatedatoms[l].jumps)[1]
  # For each allocated atom of the child process, compute the sum of the
  # observations associated with it.
  sums = map(
    x -> transpose(input.data[l]) * (state.wgroupcluslabels[l] .== x),
    Vector(1:nallocatoms),
  )

  means = @. standevs^2 * (
    (motheratommean/motheratomvar)[state.childrenatomslabels[l]] +
    sums / childrenatomvar
  )

  normals = Normal.(means, standevs)

  state.childrenallocatedatoms[l].locations[:] = rand.(normals, 1)
end

function updatechildprocessesallocjumps!(
  state::MCMCState,
  childprocess::GammaProcess,
  l,
)
  shape = state.childrenallocatedatoms[l].counter
  rate = 1 + state.auxu[l]

  gammas = Gamma.(shape, 1 / rate)

  state.childrenallocatedatoms[l].jumps[:] = rand.(gammas)
end

function updatechildprocessesallocjumps!(
  state::MCMCState,
  childprocess::GeneralizedGammaProcess,
  l,
)
  shape = state.childrenallocatedatoms[l].counter .- childprocess.sigma
  rate = childprocess.tau + state.auxu[l]

  gammas = Gamma.(shape, 1 / rate)

  state.childrenallocatedatoms[l].jumps[:] = rand.(gammas)
end

function updatechildprocessnonalloc!(
  state::MCMCState,
  model::Model,
  l,
  fkthreshold,
)
  f = x -> childprocessfkfun(x, l, model.childrenprocess, state)

  state.childrennonallocatedatoms[l].jumps = fergusonklass(f, fkthreshold)

  # For each non allocated atom of the child process, sample which atom of
  # the mother process it is associated with, using the jumps of the mother
  # process as weights.
  motheratomsjumps = getalljumps(
    state,
    group = nothing,
    motherprocess = state.groupcluslabels[l],
  )
  weights = motheratomsjumps ./ sum(motheratomsjumps)
  associations = rand(
    Categorical(weights),
    size(state.childrennonallocatedatoms[l].jumps)[1],
  )

  motheratomslocations =
    getalllocs(state, group = nothing, motherprocess = state.groupcluslabels[l])

  state.childrennonallocatedatoms[l].locations =
    samplechildloc(model, motheratomslocations[associations])

  state.childrennonallocatedatoms[l].counter =
    zeros(size(state.childrennonallocatedatoms[l].jumps)[1])
end

function childprocessfkfun(x, groupidx, childprocess::GammaProcess, state)
  return childprocess.totalmass *
         sum(
           getalljumps(
             state,
             group = nothing,
             motherprocess = state.groupcluslabels[groupidx],
           ),
         ) *
         gamma(0, x * (1 + state.auxu[groupidx]))
end

function childprocessfkfun(
  x,
  groupidx,
  childprocess::GeneralizedGammaProcess,
  state,
)
  return childprocess.totalmass / gamma(1 - childprocess.sigma) *
         sum(
           getalljumps(
             state,
             group = nothing,
             motherprocess = state.groupcluslabels[groupidx],
           ),
         ) *
         expint(
           1 + childprocess.sigma,
           x * (childprocess.tau + state.auxu[groupidx]),
         ) / x^childprocess.sigma
end

function updategroupandchildrenatomslabels!(
  input::MCMCInput,
  state::MCMCState,
  model::Model,
)
  # Updates both the groupcluslabels and the childrenatomslabels.
  g = size(state.auxu)[1]

  for l = 1:g
    oldgroupcluslabel = state.groupcluslabels[l]

    logprobs = map(
      m -> sum(
        map(
          phi -> logsumexp(
            log.(getalljumps(state, group = nothing, motherprocess = m)) +
            logpdf.(
              getatomcenterednormals(
                state,
                model,
                group = nothing,
                motherprocess = m,
              ),
              phi,
            ),
          ),
          vcat(getalllocs(state, group = l)...),
        ),
      ),
      Vector(1:model.nmotherprocesses),
    )
    logprobs .+= log.(state.probsgroupclus)
    logprobs = logprobs .- logsumexp(logprobs)
    state.groupcluslabels[l] = rand(Categorical(exp.(logprobs)))

    groupcluslabelsischanged = oldgroupcluslabel != state.groupcluslabels[l]

    if groupcluslabelsischanged
      counterstoremove = StatsBase.counts(
        state.childrenatomslabels[l],
        1:maximum(state.childrenatomslabels[l]),
      )
      #= In some cases, the maximum children atoms label in group l is lower
        than the number of allocated atoms in the corresponding mother process.
        Indeed, there could be mother atoms that are allocated only in other
        groups. Therefore, we use a for cycle because a vectorial operation
        could result in a dimension mismatch. =#
      for idx in eachindex(counterstoremove)
        state.motherallocatedatoms[oldgroupcluslabel].counter[idx] -=
          counterstoremove[idx]
      end

      # Deallocate the mother process' atoms whose counter went to zero.
      idxallocatedatom = 1
      while idxallocatedatom <=
            size(state.motherallocatedatoms[oldgroupcluslabel].counter)[1]
        if state.motherallocatedatoms[oldgroupcluslabel].counter[idxallocatedatom] ==
           0
          deallocateatom!(
            state,
            idxallocatedatom,
            group = nothing,
            motherprocess = oldgroupcluslabel,
          )
        else
          idxallocatedatom += 1
        end
      end
    end

    jumps = getalljumps(
      state,
      group = nothing,
      motherprocess = state.groupcluslabels[l],
    )
    normals = getatomcenterednormals(
      state,
      model,
      group = nothing,
      motherprocess = state.groupcluslabels[l],
    )
    nalloc = size(state.motherallocatedatoms[state.groupcluslabels[l]].jumps)[1]
    for h = 1:size(state.childrenallocatedatoms[l].locations)[1]
      nalloc = updatesingleatomlabel!(
        input,
        state,
        model,
        jumps,
        normals,
        nalloc,
        l,
        h,
        false,
        true,
        !groupcluslabelsischanged,
      )
    end
  end
end

function updatesingleatomlabel!(
  input::MCMCInput,
  state::MCMCState,
  model::Model,
  jumps,
  normals,
  nalloc,
  l,
  idx,
  wgroup,
  wasallocated,
  meaningfulpastlabel,
)
  #= Updates a single within group cluster label (wgroup==true) or a child
  process atom label (wgroup == false).
  Input:
  - jumps: all the jumps of the child (wgroup==true) or mother (wgroup==false)
    process.
  - normals: all the normals distributions derived by the child (wgroup==true)
    or mother (wgroup==false) process locations.
  - nalloc: the number of allocated child (wgroup==true) or mother
    (wgroup==false) process atoms.
  - l: the index of the group.
  - idx: the index of the children atom or of the data point.
    If the children atom is not allocated, idx must be the index w.r.t. the list
    of unallocated children atoms.
  - wgroup: look at initial description of function.
  - wasallocated: if the children atom, before this update, was already an
    allocated atom or not. If it was not allocated, the function allocates it.
    Always true when wgroup == true, there can't be a new data point.
  - meaningfulpastlabel: if the label before this update has some meaning. It
    governates if the function has to decrease the counter of the atom
    related to the past label.

  jumps and normals are modified inside the function to remain coherent with
  possible updates of the mother atoms. nalloc is an integer, therefore its
  value cannot be modified inside the function, but must be returned.
  =#

  if wgroup
    location = input.data[l][idx]
    labelsvec = state.wgroupcluslabels[l]

    # The old label was a reference to a point in the child process of group
    # l.
    group = l
    motherprocess = nothing
  else
    if wasallocated
      location = state.childrenallocatedatoms[l].locations[idx]
    else
      location = state.childrennonallocatedatoms[l].locations[idx]
    end
    labelsvec = state.childrenatomslabels[l]

    # The old label was a reference to a point in the mother process.
    group = nothing
    motherprocess = state.groupcluslabels[l]
  end

  allocatedatoms, _ =
    getatomscont(state, group = group, motherprocess = motherprocess)

  if meaningfulpastlabel
    # Save the old clustering label of the atom.
    oldidx = labelsvec[idx]

    allocatedatoms.counter[oldidx] -= 1
    # If the old label was the only one associated with a certain atom of the
    # process, remove the latter from the list of allocated atoms.
    if allocatedatoms.counter[oldidx] == 0
      deallocateatom!(
        state,
        oldidx,
        group = group,
        motherprocess = motherprocess,
      )
      nalloc -= 1
      moveinplace!(jumps, oldidx, size(jumps)[1])
      moveinplace!(normals, oldidx, size(normals)[1])
    end
  end

  logprobs = @. logpdf(normals, location) + log(jumps)
  logprobs = logprobs .- logsumexp(logprobs)

  sampledidx = rand(Categorical(exp.(logprobs)))

  if sampledidx > nalloc
    if wgroup
      # The sampled children atom is a non allocated atom. We have to sample its
      # label.
      motherjumps = getalljumps(
        state,
        group = nothing,
        motherprocess = state.groupcluslabels[l],
      )
      mothernormals = getatomcenterednormals(
        state,
        model,
        group = nothing,
        motherprocess = state.groupcluslabels[l],
      )
      mothernalloc =
        size(state.motherallocatedatoms[state.groupcluslabels[l]].jumps)[1]
      updatesingleatomlabel!(
        input,
        state,
        model,
        motherjumps,
        mothernormals,
        mothernalloc,
        l,
        sampledidx - nalloc,
        false,
        false,
        false,
      )
    else
      # The sampled atom is a new mother atom, we allocate it.
      allocateatom!(
        state,
        sampledidx - nalloc,
        group = nothing,
        motherprocess = state.groupcluslabels[l],
      )
    end
    nalloc += 1
    moveinplace!(jumps, sampledidx, nalloc)
    moveinplace!(normals, sampledidx, nalloc)
    sampledidx = nalloc
  end

  if wasallocated
    labelsvec[idx] = sampledidx
  else
    allocateatom!(state, idx, group = l)
    push!(state.childrenatomslabels[l], sampledidx)
  end
  allocatedatoms.counter[sampledidx] += 1

  return nalloc
end

function updatewgroupcluslabels!(
  input::MCMCInput,
  state::MCMCState,
  model::Model,
)
  for l = 1:input.g
    jumps = getalljumps(state, group = l)
    normals = getatomcenterednormals(state, model, group = l)
    nalloc = size(state.childrenallocatedatoms[l].jumps)[1]
    for i = 1:input.n[l]
      nalloc = updatesingleatomlabel!(
        input,
        state,
        model,
        jumps,
        normals,
        nalloc,
        l,
        i,
        true,
        true,
        true,
      )
    end
  end
end

function updateprobsgroupclus!(state::MCMCState, model::Model)
  postdirparams =
    model.dirparam * ones(model.nmotherprocesses) +
    StatsBase.counts(state.groupcluslabels, 1:model.nmotherprocesses)
  state.probsgroupclus .= rand(Dirichlet(postdirparams))
end

function updateauxu!(input::MCMCInput, state::MCMCState)
  # For each group, compute the sum of all the jumps of the corresponding
  # child process.
  sumjumps =
    map(x -> sum(x.jumps), state.childrenallocatedatoms) +
    map(x -> sum(x.jumps), state.childrennonallocatedatoms)

  #= Create g Gamma distributions, each one with shape equal to the number of
    observations in that group and rate equal to the sum of all the jumps of
    the child process of that group. We write the inverse of the second
    parameter because the Gamma() function requires the shape and the scale. =#
  gammas = @. Gamma(input.n, 1 / sumjumps)

  state.auxu[:] = rand.(gammas)
end

function updateprediction!(
  state::MCMCState,
  model::Model,
  prediction,
  grid,
  idx::Integer,
)
  g = size(prediction)[1] - 1

  # Compute the density predictions for the input groups.
  for l = 1:g
    jumps = getalljumps(state, group = l)
    jumps = jumps ./ sum(jumps)
    normals = getatomcenterednormals(state, model, group = l)
    #= Each time the function inside map is called, it returns a vector (which
      is a column vector by default) with same length as grid, then hcat puts
      together all these column in a matrix, with number of rows equal to the
      number of points in the grid and number of columns equal to the number of
      sampled locations. This matrix is multiplied with the column vector of the
      sampled jumps, to obtain a vector with length equal to the length of the
      grid. =#
    prediction[l][idx, :] = hcat(map(x -> pdf.(x, grid), normals)...) * jumps
  end

  # Compute the density predictions for a new group.
  jumps = []
  normals = []
  for m = 1:model.nmotherprocesses
    jumpstemp = getalljumps(state, group = nothing, motherprocess = m)
    jumps = vcat(jumps, @. state.probsgroupclus[m] * jumpstemp / sum(jumpstemp))
    normals = vcat(
      normals,
      Normal.(
        getprocessmeans(
          state,
          model,
          group = nothing,
          motherprocess = m,
          onlyalloc = false,
        ),
        sqrt.(
          getprocessvars(
            state,
            model,
            group = nothing,
            motherprocess = m,
            onlyalloc = false,
          ) + fill(getmixtvar(state, model), size(jumpstemp)[1]),
        ),
      ),
    )
  end
  prediction[g+1][idx, :] = hcat(map(x -> pdf.(x, grid), normals)...) * jumps
end

function hsncpmixturemodel_fit(
  input::MCMCInput,
  model::Model;
  grid = nothing,
  fkthreshold = 0.1,
  iterations = iterations,
  burnin = burnin,
  thin = thin,
)
  # If the grid is not specified, do not compute predictions.
  if grid == nothing
    prediction = nothing
  else
    prediction = [zeros(iterations, size(grid)[1]) for l = 1:(input.g+1)]
  end

  state = MCMCState(input.g, input.n, model.nmotherprocesses)
  initializemcmcstate!(input, state, model)

  output = MCMCOutput(iterations, input.g, input.n, model)

  for it in ProgressBar(1:(burnin+iterations*thin))
    updatemixtparams!(input, state, model)

    updategroupandchildrenatomslabels!(input, state, model)

    updatemotherprocesses!(state, model, fkthreshold)

    updatechildprocesses!(input, state, model, fkthreshold)

    updatewgroupcluslabels!(input, state, model)

    updateprobsgroupclus!(state, model)

    updateauxu!(input, state)

    if it > burnin && mod(it - burnin, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
      if grid != nothing
        updateprediction!(
          state,
          model,
          prediction,
          grid,
          Int((it - burnin) / thin),
        )
      end
    end
  end

  return output, prediction
end
