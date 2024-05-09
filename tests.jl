using Test
include("import.jl")

function checkwgroupcluslabelsnogaps(state::MCMCState)
  # Check that there are no gaps in the cluster labels.
  g = size(state.wgroupcluslabels)[1]

  for l = 1:g
    maxlabel = maximum(state.wgroupcluslabels[l])
    for label = 1:maxlabel
      if sum(state.wgroupcluslabels[l] .== label) == 0
        return false
      end
    end
  end
  return true
end

function checkchildrenatomslabelsnogaps(state::MCMCState)
  # Check that there are no gaps in the cluster labels.
  g = size(state.wgroupcluslabels)[1]

  maxlabel =
    maximum(map(x -> maximum(state.childrenatomslabels[x]), Vector(1:g)))

  for label = 1:maxlabel
    totcounter =
      sum(map(x -> sum(state.childrenatomslabels[x] .== label), Vector(1:g)))
    if totcounter == 0
      return false
    end
  end
  return true
end

function checkchildrencluscount(state::MCMCState)
  # Check if the saved counters are correct.
  g = size(state.wgroupcluslabels)[1]

  for l = 1:g
    maxlabel = maximum(state.wgroupcluslabels[l])
    for label = 1:maxlabel
      if sum(state.wgroupcluslabels[l] .== label) !=
         state.childrenallocatedatoms[l].counter[label]
        return false
      end
    end
  end
  return true
end

function checkmothercluscount(state::MCMCState)
  # Check if the saved counters are correct.
  g = size(state.wgroupcluslabels)[1]

  maxlabel =
    maximum(map(x -> maximum(state.childrenatomslabels[x]), Vector(1:g)))

  for label = 1:maxlabel
    totcounter =
      sum(map(x -> sum(state.childrenatomslabels[x] .== label), Vector(1:g)))
    if totcounter != state.motherallocatedatoms.counter[label]
      return false
    end
  end
  return true
end

function checkchildrenatomsontainerlength(state::MCMCState)
  # Check if the number of allocated atoms in each child process is equal to
  # the maximum label.
  g = size(state.wgroupcluslabels)[1]

  for l = 1:g
    maxlabel = maximum(state.wgroupcluslabels[l])
    if maxlabel != size(state.childrenallocatedatoms[l].counter)[1]
      return false
    end
  end
  return true
end

function checkmotheratomcontainerlength(state::MCMCState)
  # Check if the number of allocated atoms in the mother process is equal to
  # the maximum label.
  g = size(state.wgroupcluslabels)[1]

  maxlabel =
    maximum(map(x -> maximum(state.childrenatomslabels[x]), Vector(1:g)))

  return maxlabel == size(state.motherallocatedatoms.counter)[1]
end

function teststate(state::MCMCState)
  @test checkwgroupcluslabelsnogaps(state)
  @test checkchildrenatomslabelsnogaps(state)
  @test checkchildrencluscount(state)
  @test checkmothercluscount(state)
  @test checkchildrenatomsontainerlength(state)
  @test checkmotheratomcontainerlength(state)
end

@testset "MCMC" begin
  g = 3
  n = 500
  data = [rand(Normal(0, 1), n) for _ = 1:g]

  Random.seed!(7744554)

  input = MCMCInput(data)

  model = NormalMeanVarVarModel(
    mixtshape = 5,
    mixtscale = 5,
    childrentotalmass = 1,
    mothertotalmass = 1,
    motherlocsd = 6,
    motherlocshape = 5,
    motherlocscale = 5,
  )
  state = MCMCState(input.g, input.n)

  @testset "Initalization" begin
    initalizemcmcstate!(input, state, model)

    teststate(state)
  end

  @testset "Update mixture params" begin
    updatemixtparams!(input, state, model)

    teststate(state)
  end

  @testset "Update mother process" begin
    updatemotherprocess!(state, model)

    teststate(state)
  end

  @testset "Update child process" begin
    updatechildprocesses!(input, state, model)

    teststate(state)
  end

  @testset "Update child atoms labels" begin
    updatechildprocesses!(input, state, model)

    teststate(state)
  end

  @testset "Update within group labels" begin
    updatewgroupcluslabels!(input, state, model)

    teststate(state)
  end

  @testset "Update aux u" begin
    updateauxu!(input, state)

    teststate(state)
  end
end
