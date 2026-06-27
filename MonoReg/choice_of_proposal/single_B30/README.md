# single model with B=30

This folder reproduces the results in Table 5 associated with `B = 30`.

First copy the generated observed data `data_obs` and the localization results `res_SW1_precond` from the main `MonoReg` folder to this subfolder, then run the code below.

## Diagonal covariance

```bash
0..9 | ForEach-Object { python .\train_monoBP_single_init.py $_ }
0..9 | ForEach-Object { python .\train_monoBP_single_fisher.py $_ }
0..9 | ForEach-Object { python .\DebReg_fisher_crossterm.py $_ }
0..9 | ForEach-Object { python .\sampling_DebReg_fisher_crossterm.py $_ }
```

## Non-diagonal covariance

```bash
0..9 | ForEach-Object { python .\train_monoBP_single_init_nondiag.py $_ }
0..9 | ForEach-Object { python .\train_monoBP_single_fisher_nondiag.py $_ }
0..9 | ForEach-Object { python .\DebReg_fisher_crossterm_nondiag.py $_ }
0..9 | ForEach-Object { python .\sampling_DebReg_fisher_crossterm_nondiag.py $_ }
```

## Summarize results

Use `compare_nondiag_cov.ipynb` to summarize the results.
