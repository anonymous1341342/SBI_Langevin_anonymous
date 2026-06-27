# Stochastic epidemic model with 10 floors

## Generate observed data

```bash
0..49 | ForEach-Object { python .\gen_obsdata_10F.py $_ }
```

## Localization (preconditioning)

```bash
0..49 | ForEach-Object { python .\SSprecond_10F_a.py $_ }
```

## Score-matching method

```bash
0..49 | ForEach-Object { python .\scomat_SS10F.py $_ }
python .\sampling_ours_10F.py
```

## NPE

```bash
0..49 | ForEach-Object { python .\log_npe.py $_ }
python .\sampling_NPE_10F.py
```

## BSL

```bash
0..49 | ForEach-Object { python .\BSL_SI_10FSS.py $_ }
```

## ABC

```bash
0..49 | ForEach-Object { python .\SI_ABC_W1_10F.py $_ }
```

## Summarize the results

Use `res_summary.ipynb` to summarize the results.
