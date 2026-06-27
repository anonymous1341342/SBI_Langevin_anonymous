# Run each method

Run all commands from this `MonoReg` folder.

## Generate observed data

```bash
0..9 | ForEach-Object { python .\gen_obs_data.py $_ }
```

## Localization

```bash
0..9 | ForEach-Object { python .\SW1_precond.py $_ }
```

## Single model

Training and sampling are separated.

```bash
0..9 | ForEach-Object { python .\run_single_model_pipeline.py $_ }
0..9 | ForEach-Object { python .\sampling_DebReg_fisher_crossterm.py $_ }
```

## n-model

Training and sampling are separated. The scripts take a simulation budget such as `1x` or `5x`.

```bash
0..9 | ForEach-Object { python .\run_nmodel_pipeline.py $_ 1x }
0..9 | ForEach-Object { python .\sampling_nmodel_fisher.py $_ 1x }
```

## Gibbs

Sampling is run directly in one script.

```bash
0..9 | ForEach-Object { python .\sampling_gibbs.py $_ }
```

## NLE

Training and sampling are separated.

```bash
0..9 | ForEach-Object { python .\NLE_mono.py $_ }
0..9 | ForEach-Object { python .\NLE_sampling.py $_ }
```

## NPE

Training and sampling are separated.

```bash
0..9 | ForEach-Object { python .\NPE_embed_newdeepsets.py $_ }
python .\NPE_sampling.py
```

## BSL

Sampling is run directly in one script.

```bash
0..9 | ForEach-Object { python .\sampling_BSL.py $_ }
```

## ABC

Sampling is run directly in one script.

```bash
0..9 | ForEach-Object { python .\sampling_ABC_W1.py $_ }
```


# Summarize results
Use `res_summary.ipynb` to get Table 2. Use `plots.ipynb` to get Figure 1 and Figure 5. Use `compare_loss.R` to get Figure 2 and Figure 14.