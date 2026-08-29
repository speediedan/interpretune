| Stage | Preserved baseline 512x128 s | legacy 512x128 s | legacy 1024x128 s | legacy 2048x128 s | columnar_gpu 512x128 s | columnar_gpu 1024x128 s | columnar_gpu 2048x128 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 13.247 | 13.746 | 24.384 | 53.755 | 8.666 | 8.796 | 9.124 |
| feature_statistics_packaging | 0.850 | 0.834 | 1.656 | 3.161 | 0.011 | 0.019 | 0.039 |
| logits_histogram_packaging | 10.274 | 9.900 | 20.373 | 42.698 | 0.024 | 0.046 | 0.093 |
| activation_histogram_packaging | 0.142 | 0.143 | 0.301 | 1.100 | 0.045 | 0.089 | 0.163 |
| sequence_packaging | 5.212 | 5.120 | 11.085 | 22.557 | 0.296 | 0.573 | 1.196 |
| rolling_coefficient_update | 0.322 | 0.307 | 0.569 | 1.252 | 0.001 | 0.001 | 0.001 |
| batch_total | 35.224 | 35.232 | 67.649 | 141.875 | 9.156 | 10.291 | 11.433 |
