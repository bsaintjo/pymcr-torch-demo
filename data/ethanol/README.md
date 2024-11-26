# MCR-ALS Ethanol dataset

## `ethanol-df.raw-full.csv`

This represents the full dataset from the Pattern ProSpectral in raw photon AD counts. This includes spectroscopy data from ethanol from 100% to 0% (water), as well as direct light measurements at each integration time. The most raw but useful format of the dataset.

### Columns

- `dtype`: (str)
  - either "Fixed", meaning generated as part of the ProSpectral's initial calibration step where it captures spectra at fixed integration times, or "Dynamic" where the integration time was chosen based on the Pattern's proprietary integration time algorithm based on the spectra generated during the fixed calibration step
- `inttime`: (int)
  - Integration time the spectra was captured at
- `sample`: (str)
  - either `ethanol` or `light`
- `sample_idx`: (str)
  - Random UUID for keeping track of samples
- `filehash`: (str)
  - MD5 hash of ProSpectral output file, similar use case as `sample_idx`
- `conc`: (float)
  - Value between `0.0` and `100.0` representing the amount of ethanol in the mixture expressed as a percentage
- `hamamatsu_{x}`: (float)
  - Represents the number of photons captured at wavelength `x` on the near infrared Hamamatsu spectrometer.
- `avaspec_{x}`: (float)
  - Represents the number of photons captured at wavelength `x` on the visible range AvaSpec spectrometer.

## `ethanol-df.2000inttime-abs.csv`

This data was generated from `ethanol-df.raw-full.csv` by filtering for only spectra from the `Fixed` `dtype` with a `inttime` of `2000`. The spectra from the visible and near infrared at this integration time was concatenated into a single row. The data was then converted into absorbance values using the average of the `light` samples photon counts as the baseline. All columns from above are the same except the `hamamatsu_{x}` and `avaspec_{x}` columns which now represent absorbance light values.

## `ethanol-mcr.ipynb`

Jupyter notebook giving an example of loading the numpy matrices and using pyMCR to predict concentration and

## `torch-mcr.ipynb`

Same as `ethanol-mcr.ipynb` but provides a barebones example using `pytorch` to implement a model compatible with `pyMCR`. Maybe useful for evaluating `pytorch` integration into `pyMCR`.

## `ethanol-mcr.npz`

Binary, uncompressed NPZ file containing the ethanol data for D, C, and ST matrices used in pyMCR. For an example of how to use, check `ethanol-mcr.ipynb`.
Once loaded, it can be treated as a dictionary with keys representing matrices or metadata and values being the corresponding `numpy` arrays.

### Keys

- `D`: (55 samples x 2301 wavelengths)
- `C`: (55 samples x 2 components)
  - First column is percent ethanol
  - Second column is percent water
  - Sum of each row will equal 100
- `ST`: (2301 wavelengths x 2 components)
  - Represents the pure spectra of water and ethanol.
  - Calculated by taking the average absorbance values of the 0 and 100 pct ethanol samples
- `wavelengths`: (2301 wavelengths)
  - Vector of labels corresponding to visible and near infrared wavelengths
  - Derived from relevant column names from `ethanol*csv`
  - Useful for plotting
  - Format: `{spectrometer}_{wavelength}`
