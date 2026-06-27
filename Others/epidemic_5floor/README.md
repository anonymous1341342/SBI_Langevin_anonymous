# Stochastic epidemic model with 5 floors

Run all commands from this `epidemic_5floor` folder.

## Generate observed data

```bash
0..99 | ForEach-Object { python .\gen_data_obs.py $_ }
```

## Score-matching method

```bash
0..9 | ForEach-Object { python .\scomat_SI.py $_ }
python .\sampling_ours.py
```

## NPE

```bash
0..9 | ForEach-Object { python .\npe.py $_ }
python .\sampling_NPE.py
```

## BSL

```bash
0..99 | ForEach-Object { python .\SI_BSL.py $_ }
```

## ABC

```bash
0..99 | ForEach-Object { python .\SI_ABC_W1.py $_ }
```

## Summarize the results

Use `res_summary.ipynb` to summarize the results.
