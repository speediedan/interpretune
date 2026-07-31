| Stage | Preserved baseline 512x128 s | legacy 512x128 s | legacy 1024x128 s | legacy 2048x128 s | columnar_gpu 512x128 s | columnar_gpu 1024x128 s | columnar_gpu 2048x128 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 13.221 | 13.922 | 24.267 | 53.637 | 8.685 | 8.835 | 9.123 |
| feature_statistics_packaging | 0.858 | 0.848 | 1.637 | 3.158 | 0.010 | 0.019 | 0.036 |
| logits_histogram_packaging | 10.534 | 10.236 | 20.348 | 43.378 | 0.024 | 0.046 | 0.092 |
| activation_histogram_packaging | 0.141 | 0.142 | 0.297 | 1.134 | 0.045 | 0.094 | 0.165 |
| sequence_packaging | 5.200 | 5.152 | 11.068 | 22.499 | 0.326 | 0.639 | 1.262 |
| rolling_coefficient_update | 0.313 | 0.332 | 0.571 | 1.222 | 0.001 | 0.001 | 0.001 |
| batch_total | 34.842 | 36.061 | 68.106 | 142.470 | 9.784 | 10.387 | 10.947 |
