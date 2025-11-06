# Aspect Based Sentiment Analysis on Indian Airline Reviews

Dataset Used - https://www.kaggle.com/datasets/jagathratchakan/indian-airlines-customer-reviews
Model Used - https://huggingface.co/facebook/bart-large-mnli

## Installation

Install required packages with

```bash
# This is the standard command, it will automatically find your GPU
$ pip install torch torchvision torchaudio

# Then, install the rest
$ pip install pandas transformers tqdm scikit-learn
```

Optional installation without CUDA (for cpu only)

```bash
# Installs the CPU-only version of PyTorch (which transformers uses)
$ pip install torch --index-url https://download.pytorch.org/whl/cpu

# Installs all the other required libraries
$ pip install pandas transformers tqdm scikit-learn
```

# Running the model

From the repo root directory, run

```bash
$ python Final_Model/model.py
```

This will

1. pre-process the data
2. setup the model
3. do a Training/Test Set Split
4. run the model on the test set
5. save the ouutput to Output/absa_airline_results_test_set.csv
