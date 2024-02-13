# This file contains the functions to run the conditional sampler for the
# HSNCP mixture model.

function initalizemcmcstate!(state::MCMCState, model::NormalMeanModel)
  # TODO: initalization of the state of the MCMC.
end

function updateauxu!(state::MCMCState, input::MCMCInput)
  # For each group, compute the sum of all the jumps of the corresponding
  # children process.
  sumjumps =
    map(x -> sum(x.jumps), state.childrenallocatedatoms) +
    map(x -> sum(x.jumps), state.childrennonallocatedatoms)

  # Create g Gamma distributions, each one with shape equal to the number of
  # observations in that group and rate equal to the sum of all the jumps of
  # the children process of that group.
  gammas = Gamma.(input.n, sumjumps)

  state.auxu = rand.(gammas, 1)
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

    # TODO: update MCMC state

    updateauxu!(state, input)

    if it > burnin && mod(it, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
    end
  end

  return output
end
