```
    ^ ^
  <(O_o)>
   ( . )
----"-"------->>
```

# Quaia redshifts, counts and our motion

Code for the paper "The excess in the dipole of the counts of quasars is not caused by our motion" (J. Fuentes Aguilar, 2026). Every number and figure in the paper comes from these scripts.

Version 2.0 replaces version 1.0, which went with an earlier version of the paper. The scripts for f_NL are gone, `fnl.py` and `theory.py` stay only because `arf_theory.py` uses them, and the scripts for the counts, the joint test, the residual systematic, the shells and CatWISE are new.

## Data

The catalogues are public and are not included here. Put them in a folder `data/` at the top of this repository:

```
data/
  quaia_G20.0.fits
  quaia_G20.5.fits
  selection_function_NSIDE64_G20.0.fits
  selection_function_NSIDE64_G20.5.fits
  templates/selection_function_template_maps/map_*_NSIDE64.npy
  desi/QSO_NGC_clustering.dat.fits
  desi/QSO_SGC_clustering.dat.fits
  catwise/catwise_agns_masked_final_w1lt16p5_alpha.fits
  catwise/MASKS_exclude_master_final.fits
  desi_catwise/desi_catwise_like.csv
```

The Quaia catalogues, selection functions and template maps are from the Quaia release of Storey-Fisher et al. (2024), https://doi.org/10.5281/zenodo.10403370. The DESI files are the DR1 quasar clustering catalogues, in https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/. The CatWISE sample and masks are from the release of Secrest et al. (2021), https://doi.org/10.5281/zenodo.8303800.

`desi_catwise_like.csv` holds the DESI DR1 spectra of sources with AllWISE W1 < 16.6 and W1 - W2 > 0.6 at |b| > 30 deg. `sh code/catwise_desi_query.sh` downloads it from Astro Data Lab, with the query written in the script. Only `catwise_step8_desi.py` and `catwise_step9_theory_desi.py` need it.

## Running

Run everything from the top of the repository, after creating the output folders:

```
mkdir -p out/catwise

python code/arf_theory.py
python code/mocks.py
python code/results.py
python code/quaia_step1_kinematic.py
python code/quaia_step2_jackknife.py
python code/quaia_step2b_response.py
python code/quaia_step3_joint.py
python code/quaia_step3b_defence.py
python code/quaia_step4_floor.py
python code/quaia_step5b_shells.py
python code/quaia_step5c_dedoppler.py
python code/figures.py

python code/catwise_step1_reproduce.py
python code/catwise_theory_cl.py
python code/catwise_step5_wise_depth.py
sh code/catwise_desi_query.sh
python code/catwise_step8_desi.py
python code/catwise_step9_theory_desi.py
python code/catwise_step10_count_slope.py
python code/catwise_step11_half_sky.py
python code/catwise_step12_model_scan.py
```

The order matters. `mocks.py` needs the theory spectra from `arf_theory.py`, the steps of Quaia need the mocks, and `figures.py` reads the outputs of `quaia_step3_joint.py` and `quaia_step3b_defence.py`. For CatWISE, step 1 writes the mask that the others use, step 5 and step 9 need the spectra of `catwise_theory_cl.py`, step 9 needs step 8, and step 11 needs step 9. The numbers are printed to the terminal. Redirect them to a file if you want to keep them. `mocks.py` makes 200 mock skies by default, and `python code/mocks.py 50` makes fewer.

What each script gives, with the sections of the paper:

- `results.py` the dependence of the redshifts on the templates, the comparison with DESI and its tests, and the dipole of the redshifts (sections 4 and 5, appendix A). `python code/results.py 4.2` runs one part only, with the labels 3, 4.1, 4.2 and 4.3 for the errors, the templates, DESI, and the maps and the dipole.
- `mocks.py` the realistic mocks of the redshifts (section 3.3).
- `arf_theory.py` linear theory for the clustering of the redshifts (section 3.3).
- `quaia_step1_kinematic.py` the effect of the flux limit on the dipole of the redshifts (section 3.1).
- `quaia_step2_jackknife.py` the jackknife and the comparison of 370 and 800 km/s (sections 3.3 and 5).
- `quaia_step2b_response.py` the response r of the Quaia redshifts and the subsamples (sections 3.1 and 5).
- `quaia_step3_joint.py` the dipole of the counts and the test of a single velocity (sections 3.2 and 6, table 2).
- `quaia_step3b_defence.py` the joint mocks, the 64 subsets of the templates and the Ellis-Baldwin factor (section 6).
- `quaia_step4_floor.py` the residual systematic and the requirements (section 7).
- `quaia_step5b_shells.py` and `quaia_step5c_dedoppler.py` the estimators in shells of redshift (appendix B).
- `catwise*.py` the dipole of the counts of CatWISE (appendix C).
- `figures.py` figures 1 and 2, written to `figs/`.
- `sky.py`, `quaia.py`, `catwise.py`, `plotstyle.py`, `fnl.py` and `theory.py` are shared by the others.

## Python

Tested with Python 3.13 and numpy 2.5.4, scipy 1.18.1, astropy 8.0.2, healpy 1.20.1, camb 2.0.5, matplotlib 3.11.2 and pandas 3.1.0.

## Licence

MIT, see `LICENSE`. If you use the code, please cite the paper.
