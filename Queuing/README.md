# Queuing model

## Generate observed data

First generate the observed datasets:

```bash
python .\run_gen_obs_data.py
```

## Run each method

Use the commands below to run each method

### Single model

```bash
0..9 | ForEach-Object { python .\run_single_model_pipeline.py $_ }
python .\run_sampling_single.py
```

### n-model

```bash
0..9 | ForEach-Object { python .\train_nmodel_init.py $_ }
python .\run_sampling_nmodel.py
```

### NLE

```bash
0..9 | ForEach-Object { python .\NLE_001.py $_ }
0..99 | ForEach-Object { python .\NLE_001_sampling.py $_ }
```

### NPE

```bash
0..9 | ForEach-Object { python .\NPE_embed.py $_ }
python .\NPE_embed_sampling.py
```

### SNPE

```bash
0..99 | ForEach-Object { python .\NPE_embed_sequential_varybudget.py $_ }
```

### BSL

```bash
0..99 | ForEach-Object { python .\run_BSL_queuing.py $_ }
```

### ABC

```bash
0..99 | ForEach-Object { python .\run_ABC_W1.py $_ }
```


## Summarize the results
Use res_summary.ipynb to summarize the results.
