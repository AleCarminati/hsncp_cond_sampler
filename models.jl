# This file contains all the models that can be used with the conditional
# sampler.

abstract type Process end

function laplaceexp end
# Return the Laplace exponent of the process in a certain point.

@kwdef struct GammaProcess <: Process
  totalmass::Float64
end

function laplaceexp(process::GammaProcess, x)
  return process.totalmass * log(x + 1)
end

@kwdef struct GeneralizedGammaProcess <: Process
  totalmass::Float64
  sigma::Float64
  tau::Float64
end

function laplaceexp(process::GeneralizedGammaProcess, x)
  return process.totalmass / process.sigma *
         ((x + process.tau)^process.sigma - process.tau^process.sigma)
end

abstract type Model end

function samplepriormotherloc end
# Samples n locations from the prior of the mother process.

function samplechildloc end
# Samples the location of the children atoms based on its associated mother
# atoms.

function samplepriormixtparams end
# Samples from the prior of the mixture parameters.

function samplepriorgroupclusprob end
# Samples from the prior of the probability to have a certain group cluster
# label.

@kwdef struct NormalMeanModel <: Model
  #=
  		- Gaussian mixture components:
  			- fixed variance;
  			- random mean.
  		- Child processes: Gaussian kernel with fixed variance for the locations.
  		- Mother process: Gaussian with fixed variance for the locations.
  	=#

  # Standard deviation of the Gaussian mixture components.
  mixturecompsd::Float64

  # Processes of the model.
  childrenprocess::Process
  motherprocess::Process

  # Standard deviation of the Gaussian kernel.
  kernelsd::Float64

  # Hyperparameters for the the mother process location.
  motherlocmean::Float64
  motherlocsd::Float64

  # The number of mother processes.
  nmotherprocesses::Int32

  # The parameter of the Dirichlet distribution for the probability to have
  # a certain group cluster label.
  dirparam::Float64
end

function samplepriormotherloc(model::NormalMeanModel, n)
  return rand.(fill(Normal(model.motherlocmean, model.motherlocsd), n), 1)
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

function samplepriorprobgroupclus(model::Model)
  return rand(Dirichlet(model.nmotherprocesses, model.dirparam))
end

@kwdef struct NormalMeanVarModel <: Model
  #=
      - Gaussian mixture components:
        - fixed variance;
        - random mean.
      - Child processes: for the locations, Gaussian kernel with variance given
        by the second element of the mother process atom's location.
      - Mother process: for the locations, Gaussian with fixed variance and
        inverse gamma with fixed shape and scale.
    =#

  # Standard deviation of the Gaussian mixture components.
  mixturecompsd::Float64

  # Processes of the model.
  childrenprocess::Process
  motherprocess::Process

  # Hyperparameters for the the mother process location.
  motherlocmean::Float64
  motherlocsd::Float64
  motherlocshape::Float64
  motherlocscale::Float64

  # The number of mother processes.
  nmotherprocesses::Int32

  # The parameter of the Dirichlet distribution for the probability to have
  # a certain group cluster label.
  dirparam::Float64
end

function samplechildloc(model::NormalMeanVarModel, associatedmotheratomloc)
  return map(x -> rand(Normal(x[1], x[2]), 1), associatedmotheratomloc)
end

function samplepriormixtparams(model::NormalMeanVarModel)
  return []
end

@kwdef struct NormalMeanVarVarModel <: Model
  #=
      - Gaussian mixture components:
        - variance with a inverse gamma prior;
        - random mean.
      - Child processes: for the locations, Gaussian kernel with variance given
        by the second element of the mother process atom's location.
      - Mother process: for the locations, Gaussian with fixed variance and an
        inverse gamma with fixed shape and scale.
    =#

  # Hyperparameters for the Inverse Gamma prior for the variance of the mixture
  # component.
  mixtshape::Float64
  mixtscale::Float64

  # Processes of the model.
  childrenprocess::Process
  motherprocess::Process

  # Hyperparameters for the the mother process location.
  motherlocmean::Float64
  motherlocsd::Float64
  motherlocshape::Float64
  motherlocscale::Float64

  # The number of mother processes.
  nmotherprocesses::Int32

  # The parameter of the Dirichlet distribution for the probability to have
  # a certain group cluster label.
  dirparam::Float64
end

function samplepriormotherloc(
  model::Union{NormalMeanVarModel,NormalMeanVarVarModel},
  n,
)
  returnvec = [zeros(2) for _ = 1:n]
  for idx = 1:n
    returnvec[idx][1] = rand(Normal(model.motherlocmean, model.motherlocsd))
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
