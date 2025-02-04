[![Snakemake](https://img.shields.io/badge/snakemake-≥8.0.0-brightgreen.svg)](https://snakemake.github.io)
[![GitHub actions status](https://github.com/retogerber/imc_to_ims_workflow/workflows/Tests/badge.svg?branch=main)](https://github.com/retogerber/imc_to_ims_workflow/actions?query=branch%3Amain+workflow%3ATests)

# Mass Imaging Modality Integration Coregistration (MIMIC)

A snakemake workflow for the integration of Imaging Mass Spectrometry (MALDI-IMS) with Imaging Mass Cytometry (IMC).

![Overview](MIMIC_overview.png)
a) Expected experimental setup consisting of adjacent (or the same) tissue slice where IMS, IMC and microscopy slide scans (PreIMS, PreIMC, PostIMS, PostIMC) are aquired. b) Computational workflow consisting of individual a few individual of which `IMS`, `Coregistration` and `Integration` are included in this workflow. `IMC` has to be done separately (e.g. using [steinbock](https://github.com/BodenmillerGroup/steinbock)). 

## Usage

See [here](config/README.md) for detailed usage description and expected data input.

