# This file contains all the models that can be used with the conditional
# sampler.

abstract type Model end

struct NormalMeanModel <: Model
  #=
  		- Gaussian mixture components:
  			- fixed variance;
  			- random mean.
  		- Child processes: Gamma CRM with fixed total mass, for the locations we
        use a Gaussian kernel with fixed variance.
  		- Mother process: Gamma CRM with fixed total mass, for the locations we
        use a Gaussian with fixed variance.
  	=#

  # Standard deviation of the Gaussian mixture components.
  mixturecompsd::Real

  # Total mass for the Gamma CRM of the child processes.
  childrentotalmass::Real

  # Standard deviation of the Gaussian kernel.
  kernelsd::Real

  # Hyperparameters for the Gamma CRM of the mother process.
  mothertotalmass::Real
  motherlocsd::Real
end
