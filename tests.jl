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

function checkgroupcluslabelsnogaps(state::MCMCState)
  # Check that there are no gaps in group cluster labels.
  return all(StatsBase.counts(state.groupcluslabels) .> 0)
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
  for m = 1:size(state.probsgroupclus)[1]
    maxlabel = maximum(
      map(
        x -> maximum(state.childrenatomslabels[x]),
        findall(state.groupcluslabels .== m),
      ),
    )

    for label = 1:maxlabel
      totcounter = sum(
        map(
          x -> sum(state.childrenatomslabels[x] .== label),
          findall(state.groupcluslabels .== m),
        ),
      )
      if totcounter != state.motherallocatedatoms[m].counter[label]
        return false
      end
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
  for m = 1:size(state.probsgroupclus)[1]
    maxlabel = maximum(
      map(
        x -> maximum(state.childrenatomslabels[x]),
        findall(state.groupcluslabels .== m),
      ),
    )

    if maxlabel != size(state.motherallocatedatoms[m].counter)[1]
      return false
    end
  end

  return true
end

function teststate(state::MCMCState)
  @test checkwgroupcluslabelsnogaps(state)
  @test checkchildrenatomslabelsnogaps(state)
  @test checkgroupcluslabelsnogaps(state)
  @test checkchildrencluscount(state)
  @test checkmothercluscount(state)
  @test checkchildrenatomsontainerlength(state)
  @test checkmotheratomcontainerlength(state)
end

@testset "Data structures" begin
  @testset "Deallocate mother atom" begin
    g = 2
    n = 100
    data = [rand(n) for l = 1:g]
    input = MCMCInput(data)
    state = MCMCState(input.g, input.n)

    state.childrenatomslabels[1] = fill(2, 10)
    state.childrenatomslabels[2] = fill(2, 10)

    push!(state.motherallocatedatoms.locations, [3], [4])
    push!(state.motherallocatedatoms.jumps, 30, 40)
    push!(state.motherallocatedatoms.counter, 0, 4)

    deallocateatom!(state, 1, group = nothing)

    @test state.childrenatomslabels[1] == fill(1, 10)
    @test state.childrenatomslabels[2] == fill(1, 10)
    @test state.motherallocatedatoms.locations == [[4]]
    @test state.mothernonallocatedatoms.locations == [[3]]
    @test state.motherallocatedatoms.jumps == [40]
    @test state.mothernonallocatedatoms.jumps == [30]
    @test state.motherallocatedatoms.counter == [4]
    @test state.mothernonallocatedatoms.counter == [0]
  end

  @testset "Deallocate children atom" begin
    g = 2
    n = 100
    data = [rand(n) for l = 1:g]
    input = MCMCInput(data)
    state = MCMCState(input.g, input.n)

    state.wgroupcluslabels[2] .= fill(2, n)

    push!(state.childrenallocatedatoms[2].locations, [1], [2])
    push!(state.childrenallocatedatoms[2].jumps, 10, 20)
    push!(state.childrenallocatedatoms[2].counter, 0, n)

    state.childrenatomslabels[2] = [1, 2]

    push!(state.motherallocatedatoms.locations, [3], [4])
    push!(state.motherallocatedatoms.jumps, 30, 40)
    push!(state.motherallocatedatoms.counter, 1, 1)

    deallocateatom!(state, 1, group = 2)

    @test state.childrenallocatedatoms[2].locations == [[2]]
    @test state.childrennonallocatedatoms[2].locations == [[1]]
    @test state.childrenallocatedatoms[2].jumps == [20]
    @test state.childrennonallocatedatoms[2].jumps == [10]
    @test state.childrenallocatedatoms[2].counter == [n]
    @test state.wgroupcluslabels[2] == fill(1, n)
    @test state.childrenatomslabels[2] == [1]
    @test state.motherallocatedatoms.locations == [[4]]
    @test state.mothernonallocatedatoms.locations == [[3]]
    @test state.motherallocatedatoms.jumps == [40]
    @test state.mothernonallocatedatoms.jumps == [30]
    @test state.motherallocatedatoms.counter == [1]
    @test state.mothernonallocatedatoms.counter == [0]
  end

  @testset "Allocate mother atom" begin
    g = 2
    n = 100
    data = [rand(n) for l = 1:g]
    input = MCMCInput(data)
    state = MCMCState(input.g, input.n)

    push!(state.motherallocatedatoms.locations, [3])
    push!(state.motherallocatedatoms.jumps, 30)
    push!(state.motherallocatedatoms.counter, 1)
    push!(state.mothernonallocatedatoms.locations, [4])
    push!(state.mothernonallocatedatoms.jumps, 40)
    push!(state.mothernonallocatedatoms.counter, 0)

    allocateatom!(state, 1, group = nothing)

    @test state.motherallocatedatoms.locations == [[3], [4]]
    @test state.mothernonallocatedatoms.locations == []
    @test state.motherallocatedatoms.jumps == [30, 40]
    @test state.mothernonallocatedatoms.jumps == []
    @test state.motherallocatedatoms.counter == [1, 0]
    @test state.mothernonallocatedatoms.counter == []
  end

  @testset "Allocate children atom" begin
    g = 2
    n = 100
    data = [rand(n) for l = 1:g]
    input = MCMCInput(data)
    state = MCMCState(input.g, input.n)

    push!(state.motherallocatedatoms.locations, [1])
    push!(state.motherallocatedatoms.jumps, 10)
    push!(state.motherallocatedatoms.counter, 1)
    push!(state.mothernonallocatedatoms.locations, [2])
    push!(state.mothernonallocatedatoms.jumps, 20)
    push!(state.mothernonallocatedatoms.counter, 0)

    allocateatom!(state, 1, group = nothing)

    @test state.motherallocatedatoms.locations == [[1], [2]]
    @test state.mothernonallocatedatoms.locations == []
    @test state.motherallocatedatoms.jumps == [10, 20]
    @test state.mothernonallocatedatoms.jumps == []
    @test state.motherallocatedatoms.counter == [1, 0]
    @test state.mothernonallocatedatoms.counter == []
  end
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
    updatemotherprocesses!(state, model)

    teststate(state)
  end

  @testset "Update child process" begin
    updatechildprocesses!(input, state, model)

    teststate(state)
  end

  @testset "Update child atoms labels" begin
    updategroupandchildrenatomslabels!(state, model)

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
