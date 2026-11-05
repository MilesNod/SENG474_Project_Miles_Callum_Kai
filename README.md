# Fluid Prediction Model

Train a neural network to predict the next state of a lid-driven cavity fluid
simulation, then compare its predictions with the numerical solver.

Run all commands below from the repository root in PowerShell.

## Setup

If you have not already created the project environment and installed dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Train the model

```powershell
.\.venv\Scripts\python fluid-model/src/train.py
```

The script generates training data and trains for 50 epochs. After each epoch,
it saves the latest weights to `fluid-model/weights/newest_model.pt`.
Each training run starts a new model and overwrites that checkpoint.

Note: The training may take a while as it is rather graphics card heavy, we trained on a 3090 GPU

## Test the model

After training, run:

```powershell
.\.venv\Scripts\python fluid-model/src/evaluate.py
```
Please note that we will be modifying some of the code to improve the models performance prior the the evaluation assignment/milestone

Evaluation loads the saved weights and generates test trajectories using seed 4
(training uses seed 3). It reports one-step mean squared error (MSE), a baseline
that predicts no change, and errors from repeatedly using the model's own
predictions. Lower MSE is better.

To also save a plot comparing the solver, model, and absolute errors:

```powershell
.\.venv\Scripts\python fluid-model/src/evaluate.py --save-plot fluid-model/weights/evaluation.png
```

The plot is saved to `fluid-model/weights/evaluation.png`.
