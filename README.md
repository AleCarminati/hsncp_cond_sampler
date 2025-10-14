# A Conditional Sampler for the Nested Hierarchical Shot Noise Cox Process mixture models
This repository contains the code, data, and scripts accompanying the article:

> Carminati, A., Beraha, M., Camerlenghi, F., and Guglielmi, A. (2025+). **Hierarchical Shot-Noise Cox Process Mixtures for Clustering Across Groups**. Submitted.

The repository provides a complete Julia implementation of a conditional sampler for the hierarchical shot-noise Cox process (HSNCP) mixture model, together with scripts to reproduce all simulation studies and real data analysis presented in the paper.

## Repository structure

```
├── src/                  # Julia source code implementing the HSNCP sampler
├── application/          # Scripts and data for real data analysis
├── simulation1/          # Scripts for the illustrative example (Section 5.1)
├── simulation2/          # Scripts for the comparison with the HDP (Section C.1)
├── simulation3/          # Scripts for the sensitivity analysis (Section C.2)
├── simulation4/          # Scripts for the misspecified mixtures example (Section C.3)
└── README.md             # This file
```

## Software requirements

The core code is implemented in Julia (version 1.12.0).

Before running the simulation and application scripts, it is necessary to install the R package `hdp`, which implements the HDP mixture model. To do that, in an R session execute:

```R
install.packages("devtools") # Use this command if devtools is not already installed
library(devtools)
install_github("alessandrocolombi/hdp")
```

## Reproducing the results

- Illustrative example:  in a Julia session, run `include("simulation1/simulation1.jl")`.

- Comparison with the HDP: in a Julia session, run
  ```julia
  include("simulation2/simulation2_1.jl")
  include("simulation2/simulation2_2.jl")
  include("simulation2/simulation2_3.jl")
  include("simulation2/simulation2_plot.jl")
  ```
  We suggest to run the session on at most 100 threads to exploit parallelization and reduce computational times.
- Sensitivity analysis: in a Julia session, run
    ```julia
  include("simulation3/simulation3.jl")
  include("simulation3/simulation3_plot.jl")
  ```
  We suggest to run the session on at most 30 threads to exploit parallelization and reduce computational times.
- Misspecified mixture example: in a Julia session, run
    ```julia
  include("simulation4/simulation4.jl")
  include("simulation4/simulation4_plot.jl")
  ```
  We suggest to run the session on at most 30 threads to exploit parallelization and reduce computational times.
- Application: in a Julia session, run
    ```julia
  include("applicaton/application_HDP.jl")
  include("applicaton/application_HSNCP.jl")
  include("applicaton/application_plot.jl")
  ```
  In a terminal session, run
  ```bash
  Rscript application_alluvial_plot.R
  ```
## Contributing

Before committing any changes, be sure to have installed the pre-commit hooks by running the following command from the root folder:

```bash
./setup_pre_commit.sh
```

The pre-commit hooks contain some automatic checks to improve readability of the code. In particular, it uses the `JuliaFormatter` package.
