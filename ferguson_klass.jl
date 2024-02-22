# This file contains an implementation of the Ferguson and Klass algorithm
# to sample the jumps from a CRM.

function fergusonklass(func, epsilon)
  # func must be a univariate function that returns the integral of the jump
  # part of the Lévy intensity from x to +infinity, where x is func's input.
  output = Real[]
  expdist = Exponential(1)

  # The value of the last sample of the auxiliary variable.
  lastauxvar = 0

  while true
    auxvar = lastauxvar + rand(expdist)

    if lastauxvar != 0
      #= We know that the jumps are positive and that the Ferguson-Klass
       algorithm generates them in decreasing order, therefore we know that the
       new jump will be between 0 and the last jump. =#
      jump = Roots.find_zero(
        x -> func(x) - auxvar,
        (0, output[end]),
        Roots.Bisection(),
      )
    else
      # Search for a limit of the interval where to search the value of the
      # first jump.
      limitabove = 1
      if func(0) - auxvar > 0
        while func(limitabove) - auxvar > 0
          limitabove += 0.25
          if limitabove > 1000
            error("Failed to find the value of the first jump.")
          end
        end
      else
        while func(limitabove) - auxvar < 0
          limitabove += 0.25
          if limitabove > 1000
            error("Failed to find the value of the first jump.")
          end
        end
      end
      jump = Roots.find_zero(
        x -> func(x) - auxvar,
        (0, limitabove),
        Roots.Bisection(),
      )
    end

    push!(output, jump)
    lastauxvar = auxvar

    # If the last sampled jump is lower or equal than epsilon, stop the
    # algorithm.
    if output[end] <= epsilon
      break
    end
  end

  return output
end
