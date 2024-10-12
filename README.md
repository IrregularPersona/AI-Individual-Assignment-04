# Laptop Sales Prediction using Artificial Neural Networks

This assignment is aimed to predict the sales of Laptops in the month of July given the sales trends from previous months.

## Running the program

Requires the Eigen3 library. Easier to do using Linux/MVS. So if you're a windows user, either download WSL / use MVS. 
Preferably use WSL, and download Eigen and Make. Calling Make directly should compile everything all at once without an issue.

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

```
Predicted sales for month 1: 195.798
Predicted sales for month 2: 194.213
Predicted sales for month 3: 188.08
Predicted sales for month 4: 187.23
Predicted sales for month 5: 188.532
Predicted sales for month 6: 188.449
```
