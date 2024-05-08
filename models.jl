# This file contains all the models that can be used with the conditional
# sampler.

abstract type Model end

function samplepriormotherloc end
# Samples n locations from the prior of the mother process.

function samplechildloc end
# Samples the location of the children atoms based on its associated mother
# atoms.

function samplepriormixtparams end
# Samples from the prior of the mixture parameters.

abstract type GammaCRMModel <: Model end

@kwdef struct NormalMeanModel <: GammaCRMModel
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
  mixturecompsd::Float64

  # Total mass for the Gamma CRM of the child processes.
  childrentotalmass::Float64

  # Standard deviation of the Gaussian kernel.
  kernelsd::Float64

  # Hyperparameters for the Gamma CRM of the mother process.
  mothertotalmass::Float64
  motherlocsd::Float64
end

function samplepriormotherloc(model::NormalMeanModel, n)
  return rand.(fill(Normal(0, model.motherlocsd), n), 1)
end

function samplechildloc(model::NormalMeanModel, associatedmotheratomloc)
  return map(
    x -> rand(Normal(x[1], model.kernelsd), 1),
    associatedmotheratomloc,
  )
end

function samplepriormixtparams(model::NormalMeanModel)
  return []
end

@kwdef struct NormalMeanVarModel <: GammaCRMModel
  #=
      - Gaussian mixture components:
        - fixed variance;
        - random mean.
      - Child processes: Gamma CRM with fixed total mass, for the locations we
        use a Gaussian kernel with variance given by the second element of the
        mother process atom's location.
      - Mother process: Gamma CRM with fixed total mass, for the locations we
        use a Gaussian with fixed variance and an inverse gamma with fixed shape
        and scale.
    =#

  # Standard deviation of the Gaussian mixture components.
  mixturecompsd::Float64

  # Total mass for the Gamma CRM of the child processes.
  childrentotalmass::Float64

  # Hyperparameters for the Gamma CRM of the mother process.
  mothertotalmass::Float64
  motherlocsd::Float64
  motherlocshape::Float64
  motherlocscale::Float64
end

function samplechildloc(model::NormalMeanVarModel, associatedmotheratomloc)
  return map(x -> rand(Normal(x[1], x[2]), 1), associatedmotheratomloc)
end

function samplepriormixtparams(model::NormalMeanVarModel)
  return []
end

@kwdef struct NormalMeanVarVarModel <: GammaCRMModel
  #=
      - Gaussian mixture components:
        - variance with a inverse gamma prior;
        - random mean.
      - Child processes: Gamma CRM with fixed total mass, for the locations we
        use a Gaussian kernel with variance given by the second element of the
        mother process atom's location.
      - Mother process: Gamma CRM with fixed total mass, for the locations we
        use a Gaussian with fixed variance and an inverse gamma with fixed shape
        and scale.
    =#

  # Hyperparameters for the Inverse Gamma prior for the variance of the mixture
  # component.
  mixtshape::Float64
  mixtscale::Float64

  # Total mass for the Gamma CRM of the child processes.
  childrentotalmass::Float64

  # Hyperparameters for the Gamma CRM of the mother process.
  mothertotalmass::Float64
  motherlocsd::Float64
  motherlocshape::Float64
  motherlocscale::Float64
end

function samplepriormotherloc(
  model::Union{NormalMeanVarModel,NormalMeanVarVarModel},
  n,
)
  returnvec = [zeros(2) for _ = 1:n]
  for idx = 1:n
    returnvec[idx][1] = rand(Normal(0, model.motherlocsd))
    returnvec[idx][2] =
      rand(InverseGamma(model.motherlocshape, model.motherlocscale))
  end
  return returnvec
end

function samplechildloc(model::NormalMeanVarVarModel, associatedmotheratomloc)
  return map(x -> rand(Normal(x[1], x[2]), 1), associatedmotheratomloc)
end

function samplepriormixtparams(model::NormalMeanVarVarModel)
  return rand(InverseGamma(model.mixtshape, model.mixtscale), 1)
end
