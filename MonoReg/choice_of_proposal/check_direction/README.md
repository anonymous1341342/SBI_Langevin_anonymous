# Check direction

This folder contains the code for Figure 11. 

First copy the generated observed data `data_obs` and the localization results `res_SW1_precond` from the main `MonoReg` folder to this subfolder.

Then run:

```bash
python .\train_monoBP_single_init.py 0
```

Use `check_direction.ipynb` to summarize the results.
