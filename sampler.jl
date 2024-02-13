# This file contains the functions to run the conditional sampler for the
# HSNCP mixture model.

function initalizemcmcstate!(state::MCMCState)
  # TODO: initalization of the state of the MCMC.
end

function hsncpmixturemodel_fit(
  input::MCMCInput,
  model::Model;
  iterations = iterations,
  burnin = burnin,
  thin = thin,
)
  state = MCMCState(input.g, input.n)
  initalizemcmcstate!(state)

  output = MCMCOutput(iterations, input.g, input.n, model)

  for it = 1:(burnin+iterations*thin)
    # TODO: update MCMC state

    if it > burnin && mod(it, thin) == 0
      updatemcmcoutput!(input, state, output, Int((it - burnin) / thin))
    end
  end

  return output
end
