# SDEMEM real data

The procedure is similar to that in the `SDEMEM` simulation study.

The observed data are read directly from `realdata/20160427_mean_eGFP.xlsx`.

Run the following commands to train the models and obtain the posterior samples. The results can then be summarized in the notebook files. In particular, `density_plot.ipynb` is used for the posterior density plot (Figure 7 and Figure 16), `post_pred_band.ipynb` for the posterior predictive distribution plot (Figure 8), and `check_bimodal.ipynb` for the posterior predictive distribution plots at the two modes (Figure 17).

## SW1 localization

```bash
python .\SW1_realdata.py
```

## Score-matching method

```bash
python .\run_sm_pipeline.py 0
python .\sampling_ours.py
```

## NPE

```bash
python .\NPE_gen_data.py 0
python .\NPE_embed_memory_friendly_tmpfolder_newdeepsets.py 0
python .\sampling_NPE.py
```

## NLE

```bash
python .\NLE_gen_data.py 0
python .\NLE_memory_friendly.py 0
python .\sampling_NLE.py
```

## ABC

```bash
python .\sampling_ABC_W1.py
```
