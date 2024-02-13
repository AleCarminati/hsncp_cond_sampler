# This file contains an implementation of the Ferguson and Klass algorithm
# to sample the jumps from a CRM.

function fergusonklass(func, epsilon)
  # func must be a univariate function that returns the integral of the jump
  # part of the Lévy intensity from x to +infinity, where x is func's input.

  output = []
  expdist = Exponential(1)

  # The value of the last sample of the auxiliary variable.
  lastauxvar = 0

  while true
    auxvar = lastauxvar + rand(expdist, 1)
    lastauxvar = auxvar

    #= We know that the jumps are positive and that the Ferguson-Klass algorithm
    generates them in decreasing order, therefore we know that the new
    jump will be between 0 and the last jump. =#
    jump = Roots.find_zero(
      x -> func(x) - auxvar,
      (0, output[end]),
      Roots.Bisection(),
    )

    output[end] = jump

    # If the last sampled jump is lower or equal than epsilon, stop the
    # algorithm.
    if output[end] <= epsilon
      break
    end
  end
end
