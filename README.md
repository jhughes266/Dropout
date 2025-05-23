# Dropout Neural Network (Educational Project)

This project demonstrates the implementation of dropout layers within a custom neural network built from scratch using NumPy. It includes applications on both a synthetic 2D Gaussian dataset and the MNIST digit recognition dataset. The aim is to understand how dropout affects learning by preventing overfitting and improving generalization in deep neural networks.

## Table of Contents

- [Disclaimer](#disclaimer)
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
]- [My Role / What I Learned](#my-role--what-i-learned)
- [License](#license)

## Disclaimer

This project was developed as part of my personal learning. While I may have consulted various educational resources (such as tutorials, documentation, blog posts, or videos) during its creation, I do not recall all specific sources. All code has been independently written and reflects my own understanding of the topic unless explicitly stated. Any resemblance to existing material is unintentional or stems from common practices in the field.

Finally, I would also like to acknowledge that ChatGPT was used to help teach concepts and used to help debug code. When used it was used in a teaching capacity.

## Overview

This repository contains a minimal neural network framework from scratch with dropout support. The network is tested on:
- A synthetic 2D Gaussian dataset for binary classification.
- The MNIST dataset for multi-class digit classification.

Both use cases demonstrate the use of dense layers, dropout layers, and a custom mean squared error (MSE) loss function.

## Key Features

- Manual implementation of dense, dropout, and cost layers  
- Training logic using batch-based gradient descent  
- Support for tanh and sigmoid activation functions  
- Modular layer design and clear forward/backward pass logic  
- Real-time plotting of training/validation cost  
- MNIST visualization using Matplotlib  
- Demonstrates effect of dropout on performance and generalization

## Project Structure

| File / Folder       | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| ActivationFunctions | Contains tanh and sigmoid activation function implementations                   |
| DataProcessing      | Scripts for loading and generating training data (MNIST, 2D Gaussian, noise)    |
| DataSets            | Folder to store reduced MNIST dataset in CSV format                             |
| ExcelForTesting     | Optional helper files for spreadsheet-based model testing or verification       |
| Graphing            | Real-time plotting of training/validation cost via DynamicGrapher               |
| Layers              | Core layer implementations: DenseLayer, DropoutLayer, MSECostLayer              |
| PerformanceTests    | Scripts that train the network on MNIST or 2D Gaussian with/without dropout     |
| UnitTests           | (Optional) Scripts for verifying layer correctness |

## Technologies Used

| Tool   | Purpose                     |
|--------|-----------------------------|
| Python | Base language               |
| NumPy  | Matrix and tensor operations|
| Pandas | CSV file handling           |
| Matplotlib | Visualizing MNIST feature activations |

## Dataset

### 2D Gaussian

A synthetic binary classification dataset generated by the `Gauss2d` class. Two normally distributed clusters with a configurable offset and standard deviation are used to simulate class separation.

### MNIST

This project uses the **MNIST** dataset in CSV format, originally sourced from:

> [MNIST in CSV on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
> Provided by: [oddrationale](https://www.kaggle.com/oddrationale)  
> License: Public Domain

Use the following files in the `./DataSets/` directory:
- `mnist_train_reduced.csv` – A truncated version of the original training set  
- `mnist_test.csv` – Test data used in `MnistTest.py`  

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/jhughes266/Dropout.git
cd Dropout

# 2. Install required packages
pip install numpy pandas matplotlib

# 3. Ensure the following files exist:
#    - ./DataSets/mnist_train_reduced.csv
#    - ./DataSets/mnist_test.csv

# 4. Run any of the following scripts:
#    2D Gaussian classification
python PerformanceTests/2dGaussDropout.py
python PerformanceTests/2dGaussNoDropout.py

#    MNIST digit classification
python PerformanceTests/MnistWithDropout.py
python PerformanceTests/MnistBenchmark.py
```
## My Role / What I Learned

- Built a complete dropout-enabled neural network from scratch  
- Implemented gradient descent, backpropagation, and layer-wise dropout logic  
- Gained experience with data preprocessing (MNIST, synthetic)  

## License

MIT License

Copyright (c) 2025 Jotham Hughes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

