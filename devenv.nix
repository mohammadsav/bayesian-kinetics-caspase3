{ pkgs, ... }: {
  languages.python = {
    enable = true;
    package = pkgs.python313;
    manylinux.enable = true;

    venv = {
      enable = true;
      requirements = ''
        # Modeling
        pymc
        arviz
        xarray
        h5netcdf
        h5py

        # Data and math
        numpy
        polars
        pandas
        pyarrow
        scipy
        sympy

        # Plots and DAGs
        matplotlib
        plotnine
        pillow
        graphviz

        # Running the notebook
        jupyterlab
        ipykernel
      '';
    };
  };

  packages = [
    pkgs.graphviz      # dot binary, needed by the graphviz Python package
    pkgs.positron-bin
  ];

  # Point Positron at the devenv venv so it works even when launched from the desktop.
  files.".vscode/settings.json".json = {
    "python.defaultInterpreterPath" = "\${workspaceFolder}/.devenv/state/venv/bin/python";
  };
}