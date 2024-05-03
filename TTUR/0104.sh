#!/bin/bash

# less clusters
echo "0104_mnist"

# echo '3'
# python3 main_0104.py mnist 0104/3 0.0001 1.5 > ./results/0104/log_mnist_0001_15.txt
# echo '2'
# python3 main_0104.py mnist 0104/2 0.0004 1.5 > ./results/0104/log_mnist_0004_15.txt
# echo '1'
# python3 main_0104.py mnist 0104/1 0.0004 1.3 > ./results/0104/log_mnist_0004_13.txt
# echo '0'
# python3 main_0104.py mnist 0104/0 0.0002 1.0 > ./results/0104/log_mnist_0002_10.txt

# python3 main_0104_1.py mnist 0104_1/2 6 > ./results/0104_1/log_mnist_outlier5_1.txt
# python3 main_0104_1.py mnist 0104_1/2 5.5 > ./results/0104_1/log_mnist_outlier5_2.txt
# python3 main_0104_1.py mnist 0104_1/2 5 > ./results/0104_1/log_mnist_outlier5_3.txt
# python3 main_0104_1.py mnist 0104_1/2 4 > ./results/0104_1/log_mnist_outlier5_4.txt

echo '0'
python3 main_0104_1.py mnist 0104_1/3 5.5 0.6 0.05 > ./results/0104_1/log_mnist_outlier6_1.txt
echo '1'
python3 main_0104_1.py mnist 0104_1/3 5.5 0.6 0.08  > ./results/0104_1/log_mnist_outlier6_2.txt
echo '2'
python3 main_0104_1.py mnist 0104_1/3 5.5 0.5 0.08 > ./results/0104_1/log_mnist_outlier6_3.txt
echo '3'
python3 main_0104_1.py mnist 0104_1/3 5.5 0.5 0.05 > ./results/0104_1/log_mnist_outlier6_4.txt
echo '4'
python3 main_0104_1.py mnist 0104_1/3 5 0.5 0.05 > ./results/0104_1/log_mnist_outlier6_5.txt