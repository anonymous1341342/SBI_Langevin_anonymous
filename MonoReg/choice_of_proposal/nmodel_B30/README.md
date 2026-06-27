# n-model with B=30

This folder reproduces the results associated with `B = 30` in Table 4.

First copy the generated observed data `data_obs` and the localization results `res_SW1_precond` from the main `MonoReg` folder to this subfolder, then run the code below.

## Diagonal covariance

```bash
0..9 | ForEach-Object { python .\train_init.py $_ }
0..9 | ForEach-Object { python .\train_fisher.py $_ }
0..9 | ForEach-Object { python .\sampling_nmodel_fisher.py $_ }
```

## Non-diagonal covariance

```bash
0..9 | ForEach-Object { python .\train_init_nondiag.py $_ }
0..9 | ForEach-Object { python .\train_fisher_nondiag.py $_ }
0..9 | ForEach-Object { python .\sampling_nmodel_fisher_nondiag.py $_ }
```

## Summarize results

Use `compare_nondiag_cov.ipynb` to summarize the results.
