| Stage | Preserved baseline 1024x256 s | legacy 1024x256 s | legacy 2048x256 s | legacy 4096x256 s | columnar_gpu 4096x256x24576 s | columnar_gpu 4096x256x2490 s | columnar_gpu 4096x256x4096 s | columnar_gpu 1024x256 s | columnar_gpu 2048x256 s | columnar_gpu 4096x256 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 10.409 | 10.779 | 22.427 | 57.183 | 84.431 | 10.461 | 15.781 | 3.956 | 4.664 | 5.700 |
| feature_statistics_packaging | 1.494 | 1.513 | 2.893 | 5.624 | 60.679 | 5.973 | 10.001 | 0.026 | 0.049 | 0.096 |
| logits_histogram_packaging | 21.128 | 20.167 | 42.054 | 110.440 | 0.181 | 0.183 | 0.183 | 0.047 | 0.092 | 0.183 |
| activation_histogram_packaging | 0.223 | 0.222 | 0.941 | 2.262 | 62.843 | 6.345 | 10.902 | 0.114 | 0.206 | 0.512 |
| sequence_packaging | 7.584 | 7.610 | 15.805 | 36.279 | 24.113 | 4.217 | 5.436 | 0.586 | 1.154 | 2.849 |
| rolling_coefficient_update | 0.740 | 0.765 | 1.623 | 4.251 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| batch_total | 47.217 | 46.657 | 95.698 | 235.872 | 235.358 | 28.221 | 43.337 | 5.152 | 6.852 | 10.280 |
