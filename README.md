# Accelerated investigation of complex reaction cascades via digital aerosol chemistry coupled to online mass spectrometry

**Authors:** Zehua Li and S. Hessam M. Mehr

This reposistory contains the following:

* `Analysis.py`: Data analysis notebook (Marimo)
* `src/models/model_lda.py`: A simple model inspired by latent Dirichlet allocation (LDA) for decomposition of MS chromatograms into contributions of different components over time.
* Experiment code in MicroPython (targetting the [AeroBoard](https://github.com/MehrResearch/aeroboard) via the [CtrlAer](https://github.com/MehrResearch/ctrlaer) library) for experiments descibed in the manuscript and SI.

## Installation instructions
1. Clone this repository and install [`uv`](https://docs.astral.sh/uv).

```sh
git clone https://github.com/MehrResearch/aerosol_mechanism

# On Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. In the repository folder run the following to launch the notebook.

```sh
uvx marimo edit --sandbox Analysis.py
```

Once launched, the notebook will automatically look up and download the paper's dataset from Zenodo and extract it locally.
