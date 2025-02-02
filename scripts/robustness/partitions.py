import argparse

def partition_set(n, format='default'):
    """
    Generates all unique partitions of a set of size n.
    
    Args:
        n (int): The size of the set to partition.
        format (str): The format of the output ('default' or 'membership').
    
    Returns:
        list of list of list: A list containing all unique partitions.
    """
    def helper(current_set):
        """Recursively generates partitions."""
        if not current_set:
            return [[]]  # Base case: empty set has one partition - the empty set
        
        partitions = []
        # Take the first element
        first = current_set[0]
        # Generate partitions for the rest of the set
        rest_partitions = helper(current_set[1:])
        
        # For each partition of the rest, add the first element in all possible ways
        for partition in rest_partitions:
            # Add the first element to its own subset
            partitions.append([[first]] + partition)
            # Try adding the first element to existing subsets
            for i in range(len(partition)):
                new_partition = partition[:]
                new_partition[i] = [first] + new_partition[i]
                partitions.append(new_partition)
        
        return sorted(partitions)

    # Remove duplicates by converting partitions to a tuple of tuples
    from itertools import permutations
    elements = list(range(n))
    unique_partitions = {tuple(sorted(map(tuple, p))) for p in helper(elements)}
    partitions = [list(map(list, partition)) for partition in unique_partitions]
    
    if format == 'membership':
        partitions = convert_to_membership(partitions, n)
        partitions = normalize_membership(partitions)
    
    return partitions

def normalize_membership(partitions):
    """
    Normalize partitions by sorting the labels to ensure uniqueness.
    
    Args:
        partitions (list of list): The partitions in membership format.
    
    Returns:
        list of list: The normalized partitions.
    """
    normalized_partitions = []
    for partition in partitions:
        label_map = {}
        new_label = 0
        normalized_partition = []
        for label in partition:
            if label not in label_map:
                label_map[label] = new_label
                new_label += 1
            normalized_partition.append(label_map[label])
        normalized_partitions.append(normalized_partition)
    return normalized_partitions

def convert_to_membership(partitions, n):
    """
    Converts partitions to membership format.
    
    Args:
        partitions (list of list of list): The partitions to convert.
        n (int): The size of the set.
    
    Returns:
        list of list: The partitions in membership format.
    """
    membership_partitions = []
    for partition in partitions:
        membership = [-1] * n
        for subset_index, subset in enumerate(partition):
            for element in subset:
                membership[element] = subset_index
        membership_partitions.append(membership)
    return membership_partitions

def main():
    parser = argparse.ArgumentParser(description='Generate all unique partitions of a set of size n.')
    parser.add_argument('n', type=int, help='The size of the set to partition')
    parser.add_argument('--format', type=str, choices=['default', 'membership'], default='default', help='The format of the output')
    args = parser.parse_args()
    
    result = partition_set(args.n, args.format)
    print(result)

if __name__ == "__main__":
    main()