# Bayesian kinetics: caspase-3

Bayesian fits of Michaelis–Menten and Hill kinetics to caspase-3 progress curves, using PyMC. Six models of increasing complexity, from one curve with no offset to all five substrate concentrations fitted jointly, then time predictions from the integrated rate equations.

Full write-up: [My Website](https://mohammadsav.com/projects/bayesian-kinetics)

## Layout

- `data/` original `interferENZY_caspase3.txt`, the cleaned `caspase.csv`, and the long-format `kinetics_data.csv` the notebook writes
- `functions/` data loading, PyMC model builders, plotting, tables, time analysis
- `model/` saved prior and posterior samples for all six models (`.nc`)
- `notebooks/` `caspase_all_models.ipynb`, the full pipeline
- `plots/` every figure, as PNG and SVG
- `tables/` every table the write-up quotes, as CSV

## Running it
Tested on NixOS. `devenv shell` needs Nix (Linux or macOS) and a terminal. If you'd rather not use Nix, the package list in `devenv.nix` is plain pip syntax, so `pip install` it into any Python 3.13 environment and open the notebook however you like.

```
devenv shell
positron .
```
Positron opens with the devenv Python already selected. Open `notebooks/caspase_all_models.ipynb` and run the cells.

If you want to run the notebook, these are the settings to pay attention to:

```python
# True refits the models listed in MODELS_TO_RESAMPLE. False loads the saved posteriors.
RESAMPLE = False

# Which models to refit when RESAMPLE is True.
MODELS_TO_RESAMPLE = [1, 2, 3]

# True also previews every plot in the notebook. Files are saved either way.
DISPLAY_PLOTS = False
```

> [!WARNING]
> I recommend NOT resampling Model 6 unless you have read the code and have memory to spare. At 1,000 draws it climbed past 34 GB of RAM and crashed. The saved posterior was run at 100 draws and 100 tuning steps. Models 1 to 5 were fine for me, but I still recommend resampling one model at a time. You have been warned :P



## Data

Pinto, M. F., et al. (2019). A simple linearization method unveils hidden enzymatic assay interferences. *Biophysical Chemistry*, 252, 106193. https://doi.org/10.1016/j.bpc.2019.106193

Dataset: https://zenodo.org/records/3957320 (CC BY 4.0)

<details>
<summary>Dataset credit and attribution</summary>

### Creators

This project uses the dataset accompanying "A simple linearization method unveils hidden enzymatic assay interferences", created by Maria Filipa Pinto, Jorge Ripoll-Rozada, Helena Ramos, Emma E. Watson, Charlotte Franck, Richard J. Payne, Lucília Saraiva, Pedro José Barbosa Pereira, Annalisa Pastore, Fernando Rocha, and Pedro M. Martins.

The dataset is titled “Catalysis of Ac-DEVD-AMC by caspase-3 (dataset formatted for analysis by interferENZY)”.

### Copyright notice

Copyright notice: No separate copyright notice was found in the supplied dataset documentation.

### License

The source material is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The license text is available at [https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode).

### Disclaimer of warranties

Disclaimer of warranties: The source material is provided “as-is” and “as-available,” without warranties, to the extent permitted by law. See [Section 5 of the CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode#s5) for the full disclaimer of warranties and limitation of liability.

### Links to the dataset and article

Source: [Link to Dataset](https://zenodo.org/records/3957320) | [Link to Article](https://doi.org/10.1016/j.bpc.2019.106193)

### Modifications

I renamed the dataset file, removed the first five metadata lines, and added custom column names. I preserved the original tab-separated format and measurement values. I documented additional data-processing steps in the Python code in the GitHub Repo.

</details>

## License

Code and derived outputs: see `LICENSE`.
