# Accelerated investigation of complex reaction cascades via digital aerosol chemistry coupled to online mass spectrometry

Authors: Zehua Li and S. Hessam M. Mehr

This reposistory contains the following:

* `Analysis.ipynb`: Data analysis notebook
* `src/models/model_lda.py`: A simple model based on latent Dirichlet allocation (LDA) for decomposition of MS chromatograms into contributions of different components over time.
* Experiment code in MicroPython (targetting the [AeroBoard](https://github.com/MehrResearch/aeroboard) via the [CtrlAer](https://github.com/MehrResearch/ctrlaer) library) for experiments descibed in the manuscript and SI.

## Installation instructions
1. Download and extract the [paper dataset](https://zenodo.org/records/17601125/files/paper_dataset.tar.gz?download=1) from Zenodo. The following commands can be used on Linux or macOS to automate this. On Windows you can substitute `curl.exe` for `curl`, or manually download the file and use 7zip or similar to extract the `.tar.gz` archive.

```sh
# On Windows use curl.exe instead of just curl
curl -O 'https://zenodo.org/records/17601125/files/paper_dataset.tar.gz?download=1'
tar xzvf paper_dataset.tar.gz
```

2. Install [`uv`](https://docs.astral.sh/uv). This will free you from having to install Python and manually set up the project requirements without interfering with your local Python setup.

```sh
# On Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```sh
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

4. In the repository folder run the following to launch the notebook.

```sh
uvx --with-requirements pyproject.toml -p 3.12 jupyter lab
```

