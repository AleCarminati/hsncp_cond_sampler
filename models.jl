# This file contains all the models that can be used with the conditional
# sampler.

abstract type Model end

struct NormalMeanModel <: Model
  #=
  		- Gaussian mixture components:
  			- fixed variance;
  			- random mean.
  		- Child processes: Lévy intensity for the jumps of is a Gamma with fixed
  			parameters, for the locations we use a Gaussian kernel with fixed
  			variance.
  		- Mother process: Lévy intensity for the jumps is a Gamma with fixed
  			parameters, while for the locations is a Gaussian with fixed variance.

  	=#

  # Standard deviation of the Gaussian mixture components.
  mixturecompsd::Real

  # Hyperparameters for the Gamma distribution of the jumps of the children
  # processes.
  childrenjumpshape::Real
  childrenjumprate::Real

  # Standard deviation of the Gaussian kernel.
  kernelsd::Real

  # Hyperparameters for the Gamma distribution and the Gaussian distribution of
  # the jumps of the mother process.
  motherjumpshape::Real
  motherjumprate::Real
  motherlocsd::Real
end
