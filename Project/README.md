### Launch

`python main.py -lib [sklearn OR torch] -m [model.yaml]`

OR (to run experiments on every model of given lib)

`python main.py -lib [sklearn OR torch]` 

### Directory
- `checkpoints/`: saved model artifacts and configs for past runs

- `configs/`: experiment settings and model hyperparameter YAMLs

- `data/`: raw inputs and processed datasets used for training/testing

- `Doc/`: project documentation and reference PDFs

- `eda_output/`: exploratory data analysis outputs and plots

- `notebook/`: research and demo notebooks for experiments

- `results/`: aggregated metrics from completed experiments. According to **score**, **datetime** and **model_name**, easily load checkpoint in `checkpoints/`

- `src/`: source code for data loading, preprocessing, models, and training

- `models/`: local model assets cache (currently empty/placeholder)

- `todo.md`: project task list

