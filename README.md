# Solution of Phase 2: Federated learning to  predict pushback time Challenge (by Cuong_Syr team)

Username: cuongk14

## Summary

The solution is based on training collaboratively a single neural network regressor over a set of clients where each client is associated with one airline. On each round, each client will update its own parameters by stochastic gradient descent (SGD) and then send the model updates to the serve. The server aggregates the model updates and send the aggregated data to each client.

# Setup

1. Create an environment using Python 3.8. The solution was originally run on Python 3.8.12. 
```
conda create --name example-submission python=3.8
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Download the data from the competition page into `data/raw`

The structure of the directory before running training or inference should be:
```
example_submission
├── data
│   ├── processed      <- submission result folder.
│   └── raw            <- The original, immutable data dump.
├── models             <- Trained models
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── predict.py
│   ├── train.py
│   └── features.py
    └── utils.py
    └── network.py
    
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment
```

# Hardware

The solution was run on macOS Ventura, version 13.31 with Apple M1 Pro Processor. 
- Number of CPUs: 10
- Memory: 32 GB

Both training and inference were run on CPU.
- Training time: ~ 1.5 hours. Note that we need to use 2% of the whole labeled data for training. Using the whole data can crash the memory.  
- Inference time: ~30 minutes for  full data of one  aiport. .

# Run training

To run training from the command line: `python src/train.py`

By default, trained model weights will be saved to `models/federated_weights.npz`. 


# Run prediction

To run inference from the command line: `python src/predict.py`

By default, predictions will be saved out to `data/processed/submission.csv`.

--------
