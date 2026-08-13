| Stage | Preserved baseline 512x128 s | legacy 512x128 s | legacy 1024x128 s | legacy 2048x128 s | columnar_gpu 512x128 s | columnar_gpu 1024x128 s | columnar_gpu 2048x128 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 13.594 | 14.271 | 25.146 | 54.341 | 8.750 | 8.859 | 9.222 |
| feature_statistics_packaging | 0.834 | 0.823 | 1.599 | 3.173 | 0.011 | 0.019 | 0.036 |
| logits_histogram_packaging | 10.522 | 9.754 | 19.948 | 41.863 | 0.024 | 0.046 | 0.093 |
| activation_histogram_packaging | 0.146 | 0.142 | 0.297 | 1.155 | 0.047 | 0.091 | 0.171 |
| sequence_packaging | 5.432 | 5.194 | 11.172 | 22.654 | 0.337 | 0.635 | 1.291 |
| rolling_coefficient_update | 0.326 | 0.323 | 0.548 | 1.237 | 0.001 | 0.001 | 0.001 |
| batch_total | 35.908 | 35.316 | 68.351 | 142.108 | 9.750 | 10.318 | 10.900 |
