| Stage | Preserved baseline 1024x256 s | legacy 1024x256 s | legacy 2048x256 s | legacy 4096x256 s | columnar_gpu 4096x256x24576 s | columnar_gpu 4096x256x2490 s | columnar_gpu 4096x256x4096 s | columnar_gpu 1024x256 s | columnar_gpu 2048x256 s | columnar_gpu 4096x256 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 10.559 | 10.573 | 22.133 | 56.849 | 84.193 | 10.470 | 16.004 | 3.946 | 4.657 | 5.738 |
| feature_statistics_packaging | 1.494 | 1.513 | 2.918 | 5.629 | 58.577 | 6.029 | 9.931 | 0.026 | 0.049 | 0.095 |
| logits_histogram_packaging | 21.271 | 21.153 | 42.413 | 111.825 | 0.182 | 0.181 | 0.181 | 0.046 | 0.092 | 0.182 |
| activation_histogram_packaging | 0.221 | 0.218 | 0.979 | 2.322 | 62.041 | 6.750 | 10.336 | 0.116 | 0.210 | 0.507 |
| sequence_packaging | 7.571 | 7.520 | 15.684 | 35.890 | 23.041 | 4.227 | 5.465 | 0.581 | 1.159 | 2.729 |
| rolling_coefficient_update | 0.743 | 0.740 | 1.585 | 4.238 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| batch_total | 47.462 | 47.990 | 95.507 | 237.019 | 231.094 | 28.865 | 42.759 | 5.174 | 6.883 | 9.746 |
