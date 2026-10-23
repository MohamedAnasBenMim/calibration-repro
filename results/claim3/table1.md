**Test ECE (%)** (95% bootstrap CI over test samples)

| Model | Uncalibrated | Hist. Binning | Isotonic | Temp. Scaling | Vector Scaling | Matrix Scaling |
|---|---|---|---|---|---|---|
| lenet5 | 2.08 [1.66, 2.62] | 1.41 [1.08, 2.02] | 1.56 [1.30, 2.13] | 1.03 [0.81, 1.63] | 1.09 [0.79, 1.67] | 1.46 [1.13, 2.03] |
| resnet8 | 1.84 [1.46, 2.29] | 1.37 [1.09, 1.86] | 1.40 [1.16, 1.90] | 0.97 [0.72, 1.46] | 1.21 [0.90, 1.67] | 1.67 [1.37, 2.16] |

**Test Error (%)**

| Model | Uncalibrated | Hist. Binning | Isotonic | Temp. Scaling | Vector Scaling | Matrix Scaling |
|---|---|---|---|---|---|---|
| lenet5 | 9.60 | 9.70 | 9.56 | 9.60 | 9.54 | 9.60 |
| resnet8 | 7.37 | 7.49 | 7.33 | 7.37 | 7.49 | 7.71 |

**Test NLL**

| Model | Uncalibrated | Hist. Binning | Isotonic | Temp. Scaling | Vector Scaling | Matrix Scaling |
|---|---|---|---|---|---|---|
| lenet5 | 0.267 | 0.439 | 0.316 | 0.262 | 0.263 | 0.291 |
| resnet8 | 0.208 | 0.283 | 0.256 | 0.203 | 0.209 | 0.224 |
