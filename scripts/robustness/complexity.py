import time
from partitions import partition_set
from edges import generate_connected_edges

def verify_complexity(max_n):
    """
    Verifies the complexity of the partition_set function by ranging the value of n
    and checking the length of the output as n increases.
    
    Args:
        max_n (int): The maximum value of n to test.
    """
    results = []
    for n in range(1, max_n + 1):
        start_time = time.time()
        partitions = partition_set(n)
        end_time = time.time()
        duration = end_time - start_time
        results.append((n, len(partitions), duration))
    return results

def verify_edges_complexity(max_n):
    """
    Verifies the complexity of the generate_connected_edges function by ranging the value of n
    and checking the length of the output as n increases.
    
    Args:
        max_n (int): The maximum value of n to test.
    """
    results = []
    for n in range(1, max_n + 1):
        start_time = time.time()
        edges = generate_connected_edges(n)
        end_time = time.time()
        duration = end_time - start_time
        results.append((n, len(edges), duration))
        print(f"n={n}, edges={len(edges)}, time={duration:.4f}s")
    return results

if __name__ == "__main__":
    max_n = 7  # You can change this value to test larger n
    verify_complexity(max_n)
    verify_edges_complexity(max_n)
