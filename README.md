# 2D Decision

Code and raw data for:

Yul HR Kang, Anne Löffler, Danique Jeurissen, Ariel Zylberberg, Daniel M Wolpert, Michael N Shadlen, "Multiple decisions about one object involve parallel sensory acquisition but time-multiplexed evidence incorporation". Biorxiv 2020.10.15.341008 (2020) [doi:10.1101/2020.10.15.341008.](https://doi.org/10.1101/2020.10.15.341008)

Please cite the paper if you use the code and/or the data.

To use pre-computed model outputs, download files from [figshare](http://dx.doi.org/10.6084/m9.figshare.13607255) and extract each .zip file under the `data` folder (e.g., put the contents of `orig.zip` under `data/orig/`).

## Figure 2 and fits for Figure 2 supplement 1
 * All MATLAB scripts are in folder `model_fit_RT_models`: `run_all_analysis.m` will execute all code that is required to plot the serial and parallel choice-RT fits 
   * Note: By default, the `run_fit_2D()` function will not be called since fitting takes several days to complete. To re-fit the data, run `run_fit_2D.m` on a cluster (see runcode.sh), then manually move files with new fits to `from_fits` folder and then execute all other functions in `run_all_analysis.m` to create model predictions based on fit parameters
   
 (AZ wrote this part of the code)
   
 * For Fig 2 Suppl 3: To get the BF values for the binary-choice task (pink bars), go to `analysis_binChoice_exp` folder and run `run_sim_binChoice_Fig2_Suppl3.m`
 
 (AL wrote this part of the code)


## Figure 2 supplements 1-5 (fits for supplements 2-5 and plots for 1-5)
* In Python, set `Decision2D/Decision2DPy` as the working folder, and run
  * `dtb.RT.plot_model_comp_dtb_simple.py` to plot **supplement 1**
  * `dtb.RT.plot_nonparam_RT_recovery_MATLAB.py` to plot **supplements 2-5**
* To fit models anew, move parameter files in `data/Fit.D2.RT.Td2Tnd.Main` elsewhere and run `main_fig2supp2_5.m`. (This may take several days.)
* To fit with only a few iterations to see how the code runs, find `'max_iter'` in `main_fig2supp2_5.m` and change it to a small number (e.g., 1).
* Then in folder `Decision2D/Decision2DMatlab`, run `main_fig2supp2_5.m` to load and use fitted parameters to export predictions that will be used for plotting.

(YK wrote this part of the code.)

## Figure 3
* In MATLAB, go to `analysis_short_Dur_exp` folder and run `run_analysis_short_dur_data.m`

(AZ wrote this part of the code.)

## Figure 4
* In Python, set `Decision2D/Decision2DPy` as the working folder, and run `dtb.VD.dtb_2D_fit_VD.py` to load and use fitted parameters to reproduce the figure.
* To fit models anew, move parameter files in `data/Data_2D_Py/dtb/VD/` elsewhere and run `dtb_2D_fit_VD.py`. (This may take several days.)
* To fit with only a few iterations to see how the code runs, find `'max_epoch0'` in `dtb_2D_fit_VD.py` and change it to a small number (e.g., 1).

(YK wrote this part of the code.)

## Figure 4 supplements 1 and 2
* Run `Decision2D/Decision2DPy/dtb/VD/dtb_2D_recover_VD.py` to load and use fitted parameters to reproduce the figure. Note that you need files in `data/Data_2D_Py/dtb/VD/` to start this part of the analysis; otherwise the above fits will be run anew.
* To fit models anew, move parameter files in `data/Data_2D_Py/dtb/VD_model_recovery/` elsewhere and run `dtb_2D_recover_VD.py`. (This may take several days.)
* To fit with only a few iterations to see how the code runs, find `'max_epoch0'` in `dtb_2D_fit_VD.py` and change it to a small number (e.g., 1). Note that this is done the same way as for Figure 4, since both analyses use the same code for fitting.

(YK wrote this part of the code.)

## Figure 5 and Figure 5 supplement 1
* In MATLAB, go to folder `model_fit_RT_models` and run `run_fig2(2,1,0)` from command window (for re-fitting of model, see instructions above for main Fig 2) 
 
(AZ wrote this part of the code.)

## Figure 5 supplement 2
* In MATLAB, go to `model_fit_RT_models` folder and run `run_modelfree_comp_Fig5_Suppl2.m`
 
 (AL wrote this part of the code)

## Figure 6
* In MATLAB, go to `model_switching_Bimanual` folder and run `run_all_analysis.m`
 
(AZ wrote this part of the code.)

## Figure 7 and Figure 7 supplement 1
* In MATLAB, go to `analysis_binChoice_exp` folder and run `run_DDM_Fig7.m` to reproduce Fig 7, and `run_gammaRT_Fig7_Suppl1_A.m` and `run_plotModels_Fig7_Suppl1_B.m` for Fig 7 supplement 1. 
  * Note: To re-fit the gamma RT model, simply run `run_gammaRT_Fig7_Suppl1_A.m`, which saves model fits in `results_RTmodel.mat`. This takes a few minutes to run, so for Fig 7 supplement 1B, `run_plotModels_Fig7_Suppl1_B.m` does NOT re-fit the model, but simply reads saved results from the `results_RTmodel.mat` file.
 
 (AL and DW wrote this part of the code)
