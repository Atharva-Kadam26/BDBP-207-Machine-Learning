"""Write a program to partition a dataset (simulated data for regression)  into two parts, based on a feature (BP) and for a threshold, t = 80. Generate additional two partitioned datasets based on different threshold values of t = [78, 82]."""
import pandas as pd

def data_partition(filename,threshold):
    dp = pd.read_csv(filename)
    left = dp[dp['BP'] <= threshold]
    right = dp[dp['BP'] > threshold]
    return left, right

def main():
    left1,right1 = data_partition("simulated_data_multiple_linear_regression_for_ML.csv",78)
    left2,right2 = data_partition("simulated_data_multiple_linear_regression_for_ML.csv",80)
    left3,right3 = data_partition("simulated_data_multiple_linear_regression_for_ML.csv",82)
    print(f"Left dataset <= 78 = \n{left1}")
    print(f"Right dataset > 78 = \n{right1}")
    print(f"Left dataset <= 80 = \n{left2}")
    print(f"Right dataset > 80 = \n{right2}")
    print(f"Left dataset <= 82 = \n{left3}")
    print(f"Right dataset > 82 = \n{right3}")

    
if __name__ == '__main__':
    main()

