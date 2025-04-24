## Auditing Privacy of Additive Noise Mechanisms Using Linear Predictive Models
Submission to ISIT 2025

### Instructions
The `full_*.sh` files are simple scripts which submit a collection of jobs to be run in parallel. The specific script run is the `run_*.sh`. Any parameters to be changed should be changed in the `full_*.sh` script. Note that as a general rule, if parameters are set to -1, a sweep over an appropriate range is run.

#### Organization
Implementations are in `.py` files and are currently specific to a model-class and dataset pair due to differences in parameters. Datasets are defined in `mg.py` and models are defined in `models.py`.

#### Environment
The conda/mamba environment used to run these experiments is exported in `env.txt`. It can be recreated using `conda create -n statml --file env.txt`.
