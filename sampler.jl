# This file contains the functions to run the conditional sampler for the
# HSNCP mixture model.

function getatomcenterednormals(
  state::MCMCState,
  model::GammaCRMModel;
  group = nothing,
)
  #= This function returns a vector of Normal distributions.
    - group not specified: it returns Normal kernels centered in the means in
      the mother process' atoms, with variance equal to the variance in the
      mother process' atoms.
    - group specified: it returns Normal mixture components centered in the
      locations of the specified group child process' atoms. =#

  means = getprocessmeans(state, model, group = group, onlyalloc = false)
  vars = getprocessvars(state, model, group = group, onlyalloc = false)

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
  onlyalloc = true,
)
  if group == nothing
    var = model.kernelsd^2
    length = size(state.motherallocatedatoms.jumps)[1]
    if !onlyalloc
      length += size(state.mothernonallocatedatoms.jumps)[1]
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
  onlyalloc = true,
)
  if group == nothing
    vars = map(x -> x[2], state.motherallocatedatoms.locations)
    if !onlyalloc
      vars = vcat(vars, map(x -> x[2], state.mothernonallocatedatoms.locations))
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
  model::GammaCRMModel;
  group = nothing,
  onlyalloc = true,
)
  #= Function that returns a vector containing the means that are saved in the
    atoms of the mother (group = nothing) or the child (group = l) process.
    If onlyalloc = true, it returns a vector containing the means that are
    saved only in the allocated atoms.
    Note that this function only makes sense within the GammaCRMModel,
    therefore we use as input also this model to avoid using this function
    with other models. =#
  if group == nothing
    means = map(x -> x[1], state.motherallocatedatoms.locations)
    if !onlyalloc
      means =
        vcat(means, map(x -> x[1], state.mothernonallocatedatoms.locations))
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

function initalizemcmcstate!(
  input::MCMCInput,
  state::MCMCState,
  model::GammaCRMModel,
)
  state.mixtparams = samplepriormixtparams(model)

  # The number of atoms to sample for the mother process and for each child
  # process.
  mothernatoms = 10
  childnatoms = 5

  # Sample the atoms of the mother process.
  motheratomsjumps = rand(Distributions.Gamma(1, 1), mothernatoms)
  motheratomslocations = samplepriormotherloc(model, mothernatoms)

  # Contains the frequency of the sampled labels for the mother process.
  motherfreq = zeros(mothernatoms)

  for l = 1:input.g
    #= The sampled labels go from 1 to mothernatoms, while the labels of the
      state should go from 1 to the number of allocated atoms of the mother
      process. =#
    tempmotherlabels =
      rand(Distributions.DiscreteUniform(1, mothernatoms), childnatoms)

    childatomsjumps = rand(Distributions.Gamma(1, 1), childnatoms)
    # For each atom of the child process, sample its location based on
    # the associated atom of the mother process.

    childatomslocations =
      samplechildloc(model, motheratomslocations[tempmotherlabels])

    #= The sampled labels go from 1 to childnatoms, while the labels of the
      state should go from 1 to the number of allocated atoms of the child
      process. =#
    tempchildlabels =
      rand(Distributions.DiscreteUniform(1, childnatoms), input.n[l])

    # Compute the frequency of the sampled labels.
    childfreq = StatsBase.counts(tempchildlabels, childnatoms)
    # Extract the indexes of the atoms that have not been allocated.
    nonallocatomsidx = (1:childnatoms)[childfreq.==0]
    # Extract the indexes of the atoms that have been allocated.
    allocatomsidx = (1:childnatoms)[childfreq.!=0]
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
    motherfreq +=
      StatsBase.counts(tempmotherlabels[allocatomsidx], mothernatoms)
    #= Save the sampled labels in the state, only for the allocated child
      process' atoms. Attention: at this stage, the sampled labels go from 1 to
      mothernatoms, therefore they are not valid. They will be correctly
      transformed between 1 and the number of allocated mother process' atoms at
      the end of the for loop, when it is know how many atoms are of the mother
      process are allocated. =#
    state.childrenatomslabels[l] = deepcopy(tempmotherlabels[allocatomsidx])

    state.auxu[l] = rand(Distributions.Gamma(input.n[l], sum(childatomsjumps)))
  end

  # Extract the indexes of the mother process' atoms that have not been
  # allocated.
  mothernonallocatomsidx = (1:mothernatoms)[motherfreq.==0]
  # Extract the indexes of the mother process' atoms that have been allocated.
  motherallocatomsidx = (1:mothernatoms)[motherfreq.!=0]

  #= Function that transforms the sampled labels (from 1 to mothernatoms) in
    the correct cluster labels for the child processes' atoms (from 1 to the
    number of allocated atoms of the mother process).
    Then, for each group, apply the function to the sampled labels vector. =#
  obtaincluslabel = x -> findall(motherallocatomsidx .== x)[1]
  state.childrenatomslabels =
    map(l -> map(obtaincluslabel, state.childrenatomslabels[l]), 1:input.g)
  # Select the jumps and the locations of the allocated mother process' atoms.
  state.motherallocatedatoms.jumps = motheratomsjumps[motherallocatomsidx]
  state.motherallocatedatoms.locations =
    motheratomslocations[motherallocatomsidx]
  # The counters will be the frequencies of the sampled labels for the
  # allocated atoms.
  state.motherallocatedatoms.counter = motherfreq[motherallocatomsidx]

  # Select the jumps and the locations of the non allocated mother process'
  # atoms.
  state.mothernonallocatedatoms.jumps = motheratomsjumps[mothernonallocatomsidx]
  state.mothernonallocatedatoms.locations =
    motheratomslocations[mothernonallocatomsidx]
  state.mothernonallocatedatoms.counter = zeros(size(mothernonallocatomsidx)[1])
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

function updatemotherprocess!(state::MCMCState, model::Model)
  # Update the allocated atoms of the mother process.
  updatemotherprocessalloc!(state, model)
  # Update the non allocated atoms of the mother process.
  updatemotherprocessnonalloc!(state, model)
end

function updatemotherprocessalloc!(state::MCMCState, model::NormalMeanModel)
  # First, sample the jumps.
  updatemotherprocessallocjumps!(state, model)

  # Then, sample the locations.
  updatemotherprocessallocmeans!(state, model)
end

function updatemotherprocessalloc!(
  state::MCMCState,
  model::Union{NormalMeanVarModel,NormalMeanVarVarModel},
)
  # First, sample the jumps.
  updatemotherprocessallocjumps!(state, model)

  # Then, sample the locations.
  updatemotherprocessallocmeans!(state, model)

  shapes = @. model.motherlocshape + state.motherallocatedatoms.counter / 2

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
              state.motherallocatedatoms.locations[x][1]
            )^2
          )
        ),
        Vector(1:g),
      ),
    ),
    1:size(state.motherallocatedatoms.jumps)[1],
  )

  scales = @. model.motherlocscale + sums / 2

  igammas = InverseGamma.(shapes, scales)

  for i = 1:size(state.motherallocatedatoms.locations)[1]
    state.motherallocatedatoms.locations[i][2] = rand(igammas[i])
  end
end

function updatemotherprocessallocjumps!(state::MCMCState, model::GammaCRMModel)
  # Function that samples the jumps for the allocated atoms of the mother
  # process.
  g = size(state.childrenallocatedatoms)[1]
  shape = state.motherallocatedatoms.counter
  rate = 1 + model.childrentotalmass * sum(log.(1 .+ state.auxu))
  # The rate is equal for all the Gamma distributions, therefore transform it
  # from a scalar to a vector with the same values.
  rate = fill(rate, size(shape)[1])

  # We write the inverse of the second parameter because the Gamma() function
  # requires the shape and the scale.
  gammas = @. Gamma(shape, 1 / rate)

  state.motherallocatedatoms.jumps[:] = rand.(gammas)
end

function updatemotherprocessallocmeans!(state::MCMCState, model::GammaCRMModel)
  g = size(state.childrenallocatedatoms)[1]

  standevs = @. sqrt(
    1 / (
      1 / model.motherlocsd^2 +
      state.motherallocatedatoms.counter /
      $getprocessvars(state, model, group = nothing, onlyalloc = true)
    ),
  )

  # For each allocated atom j of the mother process, compute the sum of all the
  # atoms of the child processes that are associated with j.
  sums = map(
    x -> sum(
      map(
        y ->
          transpose(state.childrenatomslabels[y] .== x) *
          getprocessmeans(state, model, group = y, onlyalloc = true),
        Vector(1:g),
      ),
    ),
    1:size(state.motherallocatedatoms.jumps)[1],
  )

  means =
    @. sums / $getprocessvars(state, model, group = nothing, onlyalloc = true) *
       standevs^2

  normals = Normal.(means, standevs)

  for i = 1:size(state.motherallocatedatoms.locations)[1]
    state.motherallocatedatoms.locations[i][1] = rand(normals[i])
  end
end

function updatemotherprocessnonalloc!(state::MCMCState, model::GammaCRMModel)
  f =
    x ->
      model.mothertotalmass *
      gamma(0, x * (1 + model.childrentotalmass * sum(@. log(1 + state.auxu))))

  state.mothernonallocatedatoms.jumps = fergusonklass(f, 0.1)

  state.mothernonallocatedatoms.locations =
    samplepriormotherloc(model, size(state.mothernonallocatedatoms.jumps)[1])

  state.mothernonallocatedatoms.counter =
    zeros(size(state.mothernonallocatedatoms.jumps)[1])
end

function updatechildprocesses!(input::MCMCInput, state::MCMCState, model::Model)
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
  model::GammaCRMModel,
  l,
)

  # First, sample the jumps.
  shape = state.childrenallocatedatoms[l].counter
  rate = fill(1 + state.auxu[l], size(shape)[1])

  gammas = @. Gamma(shape, 1 / rate)

  state.childrenallocatedatoms[l].jumps[:] = rand.(gammas)

  # Then, sample the locations.
  motheratommean =
    getprocessmeans(state, model, group = nothing, onlyalloc = true)
  motheratomvar =
    getprocessvars(state, model, group = nothing, onlyalloc = true)
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

function updatechildprocessnonalloc!(state::MCMCState, model::GammaCRMModel, l)
  f = x -> model.childrentotalmass * gamma(0, x * (1 + state.auxu[l]))

  state.childrennonallocatedatoms[l].jumps = fergusonklass(f, 0.1)

  # For each non allocated atom of the child process, sample which atom of
  # the mother process it is associated with, using the jumps of the mother
  # process as weights.
  motheratomsjumps = getalljumps(state)
  weights = motheratomsjumps ./ sum(motheratomsjumps)
  associations = rand(
    Categorical(weights),
    size(state.childrennonallocatedatoms[l].jumps)[1],
  )

  motheratomslocations = getalllocs(state)

  state.childrennonallocatedatoms[l].locations =
    samplechildloc(model, motheratomslocations[associations])

  state.childrennonallocatedatoms[l].counter =
    zeros(size(state.childrennonallocatedatoms[l].jumps)[1])
end

function updatechildrenatomslabels!(state::MCMCState, model::GammaCRMModel)
  g = size(state.auxu)[1]

  jumps = getalljumps(state, group = nothing)
  normals = getatomcenterednormals(state, model, group = nothing)
  nalloc = size(state.motherallocatedatoms.jumps)[1]
  for l = 1:g
    for h = 1:size(state.childrenallocatedatoms[l].locations)[1]
      nalloc = updatesinglechildrenatomlabel!(
        state,
        model,
        jumps,
        normals,
        nalloc,
        l,
        h,
        true,
      )
    end
  end
end

function updatesinglechildrenatomlabel!(
  state::MCMCState,
  model::GammaCRMModel,
  jumps,
  normals,
  nalloc,
  l,
  h,
  wasallocated,
)
  #= Updates a single child process atom label.
  Input:
  - jumps: all the jumps of the mother process.
  - normals: all the normals distributions derived by the mother process
    locations.
  - nalloc: the number of allocated mother process atoms.
  - l: the index of the group.
  - h: the index of the children atom. If the children atom is not allocated,
    h must be the index w.r.t. the list of unallocated children atoms.
  - wasallocated: if the children atom, before this update, was already an
    allocated atom or not.

  jumps and normals are modified inside the function to remain coherent with
  possible updates of the mother atoms. nalloc is an integer, therefore its
  value cannot be modified inside the function, but must be returned.
  =#
  if wasallocated
    childatomloc = state.childrenallocatedatoms[l].locations[h]

    # Save the old clustering label of the atom.
    oldidx = state.childrenatomslabels[l][h]

    state.motherallocatedatoms.counter[oldidx] -= 1
    # If the children atom was the only one associated with a certain atom
    # of the mother process, remove the latter from the list of allocated
    # atoms.
    if state.motherallocatedatoms.counter[oldidx] == 0
      deallocateatom!(state, oldidx, group = nothing)
      nalloc -= 1
      moveinplace!(jumps, oldidx, size(jumps)[1])
      moveinplace!(normals, oldidx, size(jumps)[1])
    end
  else
    childatomloc = state.childrennonallocatedatoms[l].locations[h]
  end

  logprobs = @. logpdf(normals, childatomloc) + log(jumps)
  logprobs = logprobs .- logsumexp(logprobs)

  sampledidx = rand(Categorical(exp.(logprobs)))

  if sampledidx > nalloc
    # The sampled atom is a new atom.
    allocateatom!(state, sampledidx - nalloc, group = nothing)
    nalloc += 1
    sampledidx = nalloc
    moveinplace!(jumps, sampledidx, nalloc)
    moveinplace!(normals, sampledidx, nalloc)
  end

  if wasallocated
    state.childrenatomslabels[l][h] = sampledidx
  else
    allocateatom!(state, h, group = l)
    push!(state.childrenatomslabels[l], sampledidx)
  end
  state.motherallocatedatoms.counter[sampledidx] += 1

  return nalloc
end

function updatewgroupcluslabels!(
  state::MCMCState,
  model::GammaCRMModel,
  input::MCMCInput,
)
  for l = 1:input.g
    jumps = getalljumps(state, group = l)
    normals = getatomcenterednormals(state, model, group = l)
    nalloc = size(state.childrenallocatedatoms[l].jumps)[1]
    for i = 1:input.n[l]
      logprobs = @. logpdf(normals, input.data[l][i]) + log(jumps)
      logprobs = logprobs .- logsumexp(logprobs)

      sampledidx = rand(Categorical(exp.(logprobs)))

      # Save the old clustering label of the atom.
      oldidx = state.wgroupcluslabels[l][i]

      if sampledidx > nalloc
        # The sampled atom is a non allocated atom. We have to sample its label.
        motherjumps = getalljumps(state)
        mothernormals = getatomcenterednormals(state, model, group = nothing)
        mothernalloc = size(state.motherallocatedatoms.jumps)[1]
        updatesinglechildrenatomlabel!(
          state,
          model,
          motherjumps,
          mothernormals,
          mothernalloc,
          l,
          sampledidx - nalloc,
          false,
        )
        nalloc += 1
        sampledidx = nalloc
        moveinplace!(jumps, sampledidx, nalloc)
        moveinplace!(normals, sampledidx, nalloc)
      end
      state.wgroupcluslabels[l][i] = sampledidx
      state.childrenallocatedatoms[l].counter[sampledidx] += 1

      state.childrenallocatedatoms[l].counter[oldidx] -= 1
      # If the observation was the only one associated with a certain atom
      # of the child process, remove the latter from the list of allocated
      # atoms.
      if state.childrenallocatedatoms[l].counter[oldidx] == 0
        deallocateatom!(state, oldidx, group = l)
        nalloc -= 1
        moveinplace!(jumps, oldidx, size(jumps)[1])
        moveinplace!(normals, oldidx, size(jumps)[1])
      end
    end
  end
end

function updateauxu!(state::MCMCState, input::MCMCInput)
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
  model::GammaCRMModel,
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

  # Compute the density predictions for a new groups.
  jumps = getalljumps(state, group = nothing)
  jumps = jumps ./ sum(jumps)
  normals =
    Normal.(
      getprocessmeans(state, model, group = nothing, onlyalloc = false),
      sqrt.(
        getprocessvars(state, model, group = nothing, onlyalloc = false) +
        fill(getmixtvar(state, model), size(jumps)[1])
      ),
    )
  prediction[g+1][idx, :] = hcat(map(x -> pdf.(x, grid), normals)...) * jumps
end

function hsncpmixturemodel_fit(
  input::MCMCInput,
  model::Model;
  grid = nothing,
  iterations = iterations,
  burnin = burnin,
  thin = thin,
)
  # If not specified, initialize the grid to contain all the data points.
  if grid == nothing
    mindata = minimum([minimum(input.data[l]) for l = 1:input.g])
    maxdata = maximum([maximum(input.data[l]) for l = 1:input.g])
    grid = LinRange(mindata, maxdata, 1000)
  end
  prediction = [zeros(iterations, size(grid)[1]) for l = 1:(input.g+1)]

  state = MCMCState(input.g, input.n)
  initalizemcmcstate!(input, state, model)

  output = MCMCOutput(iterations, input.g, input.n, model)

  for it in ProgressBar(1:(burnin+iterations*thin))
    updatemixtparams!(input, state, model)

    updatemotherprocess!(state, model)

    updatechildprocesses!(input, state, model)

    updatechildrenatomslabels!(state, model)

    updatewgroupcluslabels!(state, model, input)

    updateauxu!(state, input)

    if it > burnin && mod(it - burnin, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
      updateprediction!(
        state,
        model,
        prediction,
        grid,
        Int((it - burnin) / thin),
      )
    end
  end

  return output, prediction
end
