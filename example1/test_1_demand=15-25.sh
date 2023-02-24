#!/bin/bash

echo "testing stochastic duration models on terminal 1..."
timestamp=`date +%Y-%m-%d_%H-%M-%S`
echo $timestamp

python3 main_stochastic_duration.py uniform 15 15 bi_Dirac 6 20 > results_uniform_15_15_bi_Dirac_6_20.txt
python3 main_stochastic_duration.py uniform 15 15 normal 6 20 > results_uniform_15_15_normal_6_20.txt
python3 main_stochastic_duration.py uniform 15 15 exponential 6 20 > results_uniform_15_15_exponential_6_20.txt
python3 main_stochastic_duration.py uniform 15 15 uniform 6 20 > results_uniform_15_15_uniform_6_20.txt

python3 main_stochastic_duration.py increasing 10 20 bi_Dirac 6 20 > results_increasing_10_20_bi_Dirac_6_20.txt
python3 main_stochastic_duration.py increasing 10 20 normal 6 20 > results_increasing_10_20_normal_6_20.txt
python3 main_stochastic_duration.py increasing 10 20 exponential 6 20 > results_increasing_10_20_exponential_6_20.txt
python3 main_stochastic_duration.py increasing 10 20 uniform 6 20 > results_increasing_10_20_uniform_6_20.txt

python3 main_stochastic_duration.py decreasing 10 20 bi_Dirac 6 20 > results_decreasing_10_20_bi_Dirac_6_20.txt
python3 main_stochastic_duration.py decreasing 10 20 normal 6 20 > results_decreasing_10_20_normal_6_20.txt
python3 main_stochastic_duration.py decreasing 10 20 exponential 6 20 > results_decreasing_10_20_exponential_6_20.txt
python3 main_stochastic_duration.py decreasing 10 20 uniform 6 20 > results_decreasing_10_20_uniform_6_20.txt

python3 main_stochastic_duration.py concave 10 20 bi_Dirac 6 20 > results_concave_10_20_bi_Dirac_6_20.txt
python3 main_stochastic_duration.py concave 10 20 normal 6 20 > results_concave_10_20_normal_6_20.txt
python3 main_stochastic_duration.py concave 10 20 exponential 6 20 > results_concave_10_20_exponential_6_20.txt
python3 main_stochastic_duration.py concave 10 20 uniform 6 20 > results_concave_10_20_uniform_6_20.txt

python3 main_stochastic_duration.py convex 10 20 bi_Dirac 6 20 > results_convex_10_20_bi_Dirac_6_20.txt
python3 main_stochastic_duration.py convex 10 20 normal 6 20 > results_convex_10_20_normal_6_20.txt
python3 main_stochastic_duration.py convex 10 20 exponential 6 20 > results_convex_10_20_exponential_6_20.txt
python3 main_stochastic_duration.py convex 10 20 uniform 6 20 > results_convex_10_20_uniform_6_20.txt

echo "Done!"
timestamp=`date +%Y-%m-%d_%H-%M-%S`
echo $timestamp
