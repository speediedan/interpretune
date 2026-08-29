| Stage | Preserved baseline 1024x256 s | legacy 1024x256 s | legacy 2048x256 s | legacy 4096x256 s | columnar_gpu 4096x256x24576 s | columnar_gpu 4096x256x2490 s | columnar_gpu 4096x256x4096 s | columnar_gpu 1024x256 s | columnar_gpu 2048x256 s | columnar_gpu 4096x256 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 10.158 | 10.504 | 22.189 | 57.793 | 85.330 | 10.220 | 15.731 | 3.955 | 4.512 | 5.596 |
| feature_statistics_packaging | 1.502 | 1.482 | 2.914 | 5.648 | 60.563 | 5.924 | 9.871 | 0.026 | 0.049 | 0.095 |
| logits_histogram_packaging | 21.219 | 19.825 | 42.124 | 111.839 | 0.182 | 0.183 | 0.182 | 0.046 | 0.092 | 0.182 |
| activation_histogram_packaging | 0.221 | 0.215 | 0.955 | 2.253 | 61.990 | 6.565 | 10.597 | 0.114 | 0.205 | 0.507 |
| sequence_packaging | 7.617 | 7.573 | 15.752 | 36.622 | 23.081 | 4.156 | 5.371 | 0.564 | 1.122 | 2.676 |
| rolling_coefficient_update | 0.716 | 0.713 | 1.610 | 4.247 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| batch_total | 47.150 | 46.041 | 95.037 | 238.529 | 234.291 | 28.153 | 42.571 | 5.144 | 6.839 | 9.701 |
