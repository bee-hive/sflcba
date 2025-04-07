# saft_figuren

Code coresponding to the segmentation free live-cell behavioral analysis (SF-LCBA) manuscript.

### Installing dependencies

To reproduce this code, we recommend intalling all dependencies into a miniforge conda environment using the instructions as follows.

```
conda create --name saft python=3.11.4
conda activate saft
pip install -r requirements.txt
```

Then install the `sf-lcba` package in editable mode

```
pip install -e .
```
This command installs the sflcba package as a development (editable) package, meaning that any changes you make to the code in the sflcba folder will be immediately reflected without needing to reinstall the package.

After installing these dependencies, you should be able to lauch jupyter from the `saft` conda environment and run the corresponding notebooks in this repository.
