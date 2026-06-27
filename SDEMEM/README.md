# SDEMEM

Run all commands from this `SDEMEM` folder.

## Generate observed data

```bash
0..99 | ForEach-Object { python .\gen_obs.py $_ }
```

## SW1 localization

```bash
0..9 | ForEach-Object { python .\SW1_localization.py $_ }
```

## Score-matching method

```bash
0..9 | ForEach-Object { python .\run_sm_pipeline.py $_ }
0..99 | ForEach-Object { python .\sampling_ours.py $_ }
```

## NPE

For NPE, first generate the training data, then train the 10 models, then sample for all 100 observed datasets.
Here we do not use the simple wrapped-up `sbi` training functions. In this example, those functions create additional copies of the training data when cleaning the data and splitting the train/validation sets, which can increase memory pressure and lead to out-of-memory issues. Instead, we handle the data manually and implement the training loop following the `sbi` package guidelines.

```bash
0..9 | ForEach-Object { python .\NPE_gen_data.py $_ }
0..9 | ForEach-Object { python .\NPE_embed_memory_friendly_tmpfolder_newdeepsets.py $_ }
python .\NPE_sampling.py
```

## NLE

For NLE, first generate the training data, then train the 10 models, then sample for all 100 observed datasets.
As in NPE, we do not use the simple wrapped-up `sbi` training functions here. Instead, we manually manage the data and write the training loop following the `sbi` package guidelines.

```bash
0..9 | ForEach-Object { python .\NLE_gen_data.py $_ }
0..9 | ForEach-Object { python .\NLE_memory_friendly.py $_ }
0..99 | ForEach-Object { python .\NLE_sampling.py $_ }
```

## ABC

ABC sampling runs directly on each observed dataset.

```bash
0..99 | ForEach-Object { python .\sampling_ABC_W1.py $_ }
```



## BSL

`sampling_BSL_plain_corr.py` currently uses the `task_id` and `chain_id` set inside the script, and then runs one chain.

```bash
python .\sampling_BSL_plain_corr.py
```
