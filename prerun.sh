# Create directories to store results
mkdir -p data
mkdir -p plots/eda
mkdir -p plots/sl_results
mkdir -p plots/ssl_results
mkdir -p saved_models/sl_models
mkdir -p saved_models/ssl_models

# Get data from UCIML repo
wget https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data -P data/
