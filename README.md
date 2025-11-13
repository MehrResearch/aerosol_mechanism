# Accelerated investigation of complex reaction cascades via digital aerosol chemistry coupled to online mass spectrometry

Authors: Zehua Li and S. Hessam M. Mehr

This reposistory contains the following:

* `Analysis.ipynb`: Data analysis notebook
* `src/models/model_lda.py`: A simple model based on latent Dirichlet allocation (LDA) for decomposition of MS chromatograms into contributions of different components over time.
* Experiment code in MicroPython (targetting the [AeroBoard](https://github.com/MehrResearch/aeroboard) via the [CtrlAer](https://github.com/MehrResearch/ctrlaer) library) for experiments descibed in the manuscript and SI.

## Installation instructions
1. Download and extract the [paper dataset](https://zenodo.org/records/17601125/files/paper_dataset.tar.gz?download=1) from Zenodo. The following commands can be used on Linux or macOS to automate this.

```sh
curl -O 'https://zenodo.org/records/17601125/files/paper_dataset.tar.gz?download=1'
tar xzvf paper_dataset.tar.gz
```

2. Install [`uv`](https://docs.astral.sh/uv).

3. In the repository folder run the following to launch the notebook.

```sh
uvx --with-requirements pyproject.toml -p 3.12 jupyter lab
```
