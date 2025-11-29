# mini-hwks
A lightweight experimental framework for building Hawkes-process–driven models for short-horizon price prediction.

## Overview
This repository contains the full code and artifacts used in developing a combined microstructure–Hawkes-process prediction pipeline. All core Hawkes-related methods are implemented directly inside the **methods section** of the main notebook, including intensity formulation, compensated arrivals, feature construction, and model evaluation windows.

The repo also includes the **final exported models** used for price-direction and volatility forecasting. In out-of-sample tests, the price-direction models reached roughly **70% accuracy** on short time horizons, depending on the day and regime.

## Repository Contents
- **notebooks.ipynb** – main development notebook with the Hawkes methods, feature engineering stages, and evaluation workflow  
- **mode/** – final trained models (serialized) used in the prediction pipeline  
- **Project_Hawkes.pdf** – detailed project write-up, including goals, methodology, mathematical formulation, and evaluation metrics  
- **.gitignore** – excludes all large intermediary data outputs from version control  

## Notes
This repository does **not** include raw market data, intermediate datasets, or any generated data directories. These are intentionally excluded due to size constraints and ip issues.






