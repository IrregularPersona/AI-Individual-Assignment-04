# Laptop Sales Prediction using Artificial Neural Networks

This assignment is aimed to predict the sales of Laptops in the month of July given the sales trends from previous months.

## Running the program

### FOR C++
Requires the Eigen3 library. Easier to do using Linux/MVS. So if you're a windows user, either download WSL / use MVS. 
Preferably use WSL, and download Eigen and Make. Calling Make directly should compile everything all at once without an issue.

### FOR Python
Requires Numpy.

### FOR Rust
Requires ndarray.

## Dataset

Here is the dataset used for the prediction:

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 2023 | 205 | 196 | 141 | 112 | 163 | 242 | 816 | 1043| 950 | 529 | 308 | 214 |
| 2024 | 176 | 146 | 109 | 110 | 135 | 160 | ... | ... | ... | ... | ... | ... |

## Purpose

Given said dataset, we need to predict the last 6 months of sales.

## Result of the Program

Here is the training logs of the ANN (Logging is commented out by default):

```
Epoch 0   | Learning Rate: 0.01 | Predicted (denormalized): 728.565
Epoch 100 | Learning Rate: 0.01 | Predicted (denormalized): 631.951
Epoch 200 | Learning Rate: 0.005 | Predicted (denormalized): 548.123
Epoch 300 | Learning Rate: 0.005 | Predicted (denormalized): 512.36
Epoch 400 | Learning Rate: 0.005 | Predicted (denormalized): 480.796
Epoch 500 | Learning Rate: 0.001 | Predicted (denormalized): 453.297
Epoch 600 | Learning Rate: 0.001 | Predicted (denormalized): 448.179
Epoch 700 | Learning Rate: 0.001 | Predicted (denormalized): 443.193
Epoch 800 | Learning Rate: 0.001 | Predicted (denormalized): 438.338
Epoch 900 | Learning Rate: 0.001 | Predicted (denormalized): 433.608
Epoch 0   | Learning Rate: 0.01 | Predicted (denormalized): 426.878
Epoch 100 | Learning Rate: 0.01 | Predicted (denormalized): 385.633
Epoch 200 | Learning Rate: 0.005 | Predicted (denormalized): 353.349
Epoch 300 | Learning Rate: 0.005 | Predicted (denormalized): 339.696
Epoch 400 | Learning Rate: 0.005 | Predicted (denormalized): 327.445
Epoch 500 | Learning Rate: 0.001 | Predicted (denormalized): 316.491
Epoch 600 | Learning Rate: 0.001 | Predicted (denormalized): 314.413
Epoch 700 | Learning Rate: 0.001 | Predicted (denormalized): 312.377
Epoch 800 | Learning Rate: 0.001 | Predicted (denormalized): 310.38
Epoch 900 | Learning Rate: 0.001 | Predicted (denormalized): 308.422
Epoch 0   | Learning Rate: 0.01 | Predicted (denormalized): 308.746
Epoch 100 | Learning Rate: 0.01 | Predicted (denormalized): 289.841
Epoch 200 | Learning Rate: 0.005 | Predicted (denormalized): 273.954
Epoch 300 | Learning Rate: 0.005 | Predicted (denormalized): 266.876
Epoch 400 | Learning Rate: 0.005 | Predicted (denormalized): 260.315
Epoch 500 | Learning Rate: 0.001 | Predicted (denormalized): 254.267
Epoch 600 | Learning Rate: 0.001 | Predicted (denormalized): 253.099
Epoch 700 | Learning Rate: 0.001 | Predicted (denormalized): 251.947
Epoch 800 | Learning Rate: 0.001 | Predicted (denormalized): 250.812
Epoch 900 | Learning Rate: 0.001 | Predicted (denormalized): 249.692
Epoch 0   | Learning Rate: 0.01 | Predicted (denormalized): 254.962
Epoch 100 | Learning Rate: 0.01 | Predicted (denormalized): 244.346
Epoch 200 | Learning Rate: 0.005 | Predicted (denormalized): 235.053
Epoch 300 | Learning Rate: 0.005 | Predicted (denormalized): 230.792
Epoch 400 | Learning Rate: 0.005 | Predicted (denormalized): 226.775
Epoch 500 | Learning Rate: 0.001 | Predicted (denormalized): 223.01
Epoch 600 | Learning Rate: 0.001 | Predicted (denormalized): 222.276
Epoch 700 | Learning Rate: 0.001 | Predicted (denormalized): 221.55
Epoch 800 | Learning Rate: 0.001 | Predicted (denormalized): 220.832
Epoch 900 | Learning Rate: 0.001 | Predicted (denormalized): 220.121
Epoch 0   | Learning Rate: 0.01 | Predicted (denormalized): 216.537
Epoch 100 | Learning Rate: 0.01 | Predicted (denormalized): 211.347
Epoch 200 | Learning Rate: 0.005 | Predicted (denormalized): 206.663
Epoch 300 | Learning Rate: 0.005 | Predicted (denormalized): 204.471
Epoch 400 | Learning Rate: 0.005 | Predicted (denormalized): 202.376
Epoch 500 | Learning Rate: 0.001 | Predicted (denormalized): 200.389
Epoch 600 | Learning Rate: 0.001 | Predicted (denormalized): 199.999
Epoch 700 | Learning Rate: 0.001 | Predicted (denormalized): 199.613
Epoch 800 | Learning Rate: 0.001 | Predicted (denormalized): 199.229
Epoch 900 | Learning Rate: 0.001 | Predicted (denormalized): 198.849
Epoch 0   | Learning Rate: 0.01 | Predicted (denormalized): 205.616
Epoch 100 | Learning Rate: 0.01 | Predicted (denormalized): 202.705
Epoch 200 | Learning Rate: 0.005 | Predicted (denormalized): 200.044
Epoch 300 | Learning Rate: 0.005 | Predicted (denormalized): 198.788
Epoch 400 | Learning Rate: 0.005 | Predicted (denormalized): 197.582
Epoch 500 | Learning Rate: 0.001 | Predicted (denormalized): 196.432
Epoch 600 | Learning Rate: 0.001 | Predicted (denormalized): 196.206
Epoch 700 | Learning Rate: 0.001 | Predicted (denormalized): 195.981
Epoch 800 | Learning Rate: 0.001 | Predicted (denormalized): 195.759
Epoch 900 | Learning Rate: 0.001 | Predicted (denormalized): 195.538
```

Here are the results for the next 6 months (predicted + denormalized):

### C++
```
Predicted sales for month 1: 195.798
Predicted sales for month 2: 194.213
Predicted sales for month 3: 188.08
Predicted sales for month 4: 187.23
Predicted sales for month 5: 188.532
Predicted sales for month 6: 188.449
```

### Python
```
Predicted sales for month 1: 169.34533976203215
Predicted sales for month 2: 176.61207955833908
Predicted sales for month 3: 187.18853445697857
Predicted sales for month 4: 197.21303494117075
Predicted sales for month 5: 202.51877700473221
Predicted sales for month 6: 205.96950227334116
```

### Rust
```
Predictions for the next 6 months:
Month 1: 167.48
Month 2: 170.28
Month 3: 175.82
Month 4: 178.25
Month 5: 178.43
Month 6: 178.17
```

##### Note
The Rust variant uses a dynamic RNG generator, thus causing the model to throw different values everytime.

# Answering the questions:

### Question F: Why the result of the prediction is accurate or not accurate? Explain briefly.
Because, given the dataset and the training, the resulting data seems to be in-trend with the results, given that this is a prediction.
