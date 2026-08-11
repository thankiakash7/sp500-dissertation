# Remaining Dissertation Notebooks

Copy the `notebooks/` files into your existing project's `notebooks/` folder.

## Run order

1. `06_crsp_eda_and_tests.ipynb`
2. `07_garch_baseline.ipynb`
3. `08_walk_forward_evaluation.ipynb`
4. `09_lstm_and_gru.ipynb`
5. `10_tft_model.ipynb`
6. `11_rmt_100_200_300.ipynb`
7. `12_model_comparison_dm_mcs.ipynb`
8. `13_portfolio_backtest.ipynb`

## Install the new packages

From the project root:

```bash
conda activate dissertation
python -m pip install -r requirements_next_stage.txt
```

Restart the VS Code notebook kernel after installation.

## Important

- Run one notebook at a time.
- Keep `QUICK_MODE = True` until the notebook works.
- Do not run the full 100/200/300 rolling RMT experiment immediately.
- Do not claim the static LSTM/GRU/TFT results are final walk-forward results.
- Confirm the final return target and GARCH mean specification with the supervisor.
- Never upload licensed CRSP data to a public repository.
