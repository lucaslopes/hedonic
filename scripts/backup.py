import numpy as np

def generate_probability_array(length):
  powers = np.arange(length-1, -1, -1) # [length-1, length-2, ..., 0]
  result = 1 / np.power(2, powers) # [1/2^length-1, 1/2^length-2, ..., 1/2^0]
  return result # return probability array

def sequential_probability_ratio_test(func, error=0.01, z=1.96):
  samples = []
  while True:
    sample = func()
    samples.append(sample)
    n = len(samples)
    mean = np.mean(samples)
    se = np.sqrt((mean * (1 - mean)) / n)
    ci = z * se
    if ci < error:
      break
  return mean
  # z = 1.96 # 95% confidence interval
  # z = 1.645 # 90% confidence interval
  # error = 5/100 # 5% error
  # se = np.sqrt((mean * (1 - mean)) / n)
  # ci = z * se
  # if ci < error:
  #     break