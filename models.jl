# This file contains all the models that can be used with the conditional
# sampler.

abstract type Model end

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
  mixturecompsd::Real

  # Total mass for the Gamma CRM of the child processes.
  childrentotalmass::Real

  # Standard deviation of the Gaussian kernel.
  kernelsd::Real

  # Hyperparameters for the Gamma CRM of the mother process.
  mothertotalmass::Real
  motherlocsd::Real
end

function samplepriormotherloc(model::NormalMeanModel, n)
  # Function that samples n locations from the prior of the mother process.
  return rand.(fill(Normal(0, model.motherlocsd), n), 1)
end

function samplechildloc(model::NormalMeanModel, associatedmotheratomloc)
  # Function that samples the location of the children atoms based on its
  # associated mother atoms.
  return map(
    x -> rand(Normal(x[1], model.kernelsd), 1),
    associatedmotheratomloc,
  )
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
  mixturecompsd::Real

  # Total mass for the Gamma CRM of the child processes.
  childrentotalmass::Real

  # Hyperparameters for the Gamma CRM of the mother process.
  mothertotalmass::Real
  motherlocsd::Real
  motherlocshape::Real
  motherlocscale::Real
end

function samplepriormotherloc(model::NormalMeanVarModel, n)
  # Function that samples n locations from the prior of the mother process.
  return map(
    x -> rand.(x),
    fill(
      [
        Normal(0, model.motherlocsd),
        InverseGamma(model.motherlocshape, model.motherlocscale),
      ],
      n,
    ),
  )
end

function samplechildloc(model::NormalMeanVarModel, associatedmotheratomloc)
  # Function that samples the location of the children atoms based on its
  # associated mother atoms.
  return map(x -> rand(Normal(x[1], x[2]), 1), associatedmotheratomloc)
end
