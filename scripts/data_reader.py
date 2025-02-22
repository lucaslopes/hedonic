"""Module for reading and processing data."""

import os
import re
import time
import json
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
from utils import sort_files


def collect_paths(leaf_folder: str, extension: str = ".json") -> list[str]:
    """
    Collects files with the given extension from a given leaf folder.
    This function assumes that the given folder does not contain subdirectories.
    By default, it collects .json files; you can change the extension (e.g. to ".txt") as needed.
    """
    file_paths = []
    try:
        for file in os.listdir(leaf_folder):
            if file.endswith(extension):
                full_path = os.path.join(leaf_folder, file)
                if os.path.isfile(full_path):
                    file_paths.append(full_path)
    except Exception as e:
        print(f"Error processing folder {leaf_folder}: {e}")
    return file_paths


def get_leaf_subdirs_from_subroot(subroot: str) -> list[str]:
    """
    Walks a subtree starting at subroot and returns all leaf directories.
    """
    leaf_dirs = []
    for dirpath, dirs, _ in os.walk(subroot):
        if not dirs:  # This is a leaf directory.
            leaf_dirs.append(dirpath)
    return leaf_dirs


def get_subroots(root: str, max_depth: int = 1) -> list[str]:
    """
    Returns a list of subdirectories from the root at a given depth.
    If max_depth is 1, you get the immediate child directories.
    Setting max_depth higher will go further down.
    """
    subroots = []
    # Walk the tree and measure depth relative to root.
    for dirpath, dirs, _ in os.walk(root):
        # Calculate relative depth (number of separators after the root)
        rel_depth = dirpath[len(root):].count(os.sep)
        if rel_depth == max_depth:
            subroots.append(dirpath)
            # Prevent descending further from this subroot.
            dirs[:] = []
    return subroots


def get_leaf_subdirs_parallel(root: str, subroot_depth: int = 1) -> list[str]:
    """
    Uses parallel processing to collect all leaf directories.
    First, it gathers subroots at the specified depth, then processes each
    subtree in parallel, ensuring that the returned paths are complete.
    """
    # Get subroots at the chosen depth. If none found, fall back to using the root.
    subroots = get_subroots(root, max_depth=subroot_depth)
    if not subroots:
        subroots = [root]
    with ProcessPoolExecutor() as executor:  # Process each subroot concurrently.
        results = list(executor.map(get_leaf_subdirs_from_subroot, subroots))
    leaf_dirs = [leaf for sublist in results for leaf in sublist]  # Flatten the list of lists.
    return leaf_dirs


def collect_paths_helper(args: tuple[str, str]) -> list[str]:
    """
    Helper function for parallel collection of paths.
    """
    leaf_folder, extension = args
    return collect_paths(leaf_folder, extension)


def get_paths_sorted(folder_path: str, extension: str = ".json") -> list[str]:
    """
    Gets a list of files with the given extension sorted by network seed and experiment settings,
    processing only leaf directories in parallel.
    """
    leaf_dirs = get_leaf_subdirs_parallel(folder_path, subroot_depth=2)
    args_list = [(lf, extension) for lf in leaf_dirs]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(collect_paths_helper, args_list),
            total=len(leaf_dirs),
            desc=f"Collecting {extension} paths"
        ))
    file_paths = [file for sublist in results for file in sublist]
    file_paths = sort_files(file_paths)
    return file_paths



def split_json_paths(json_paths: list[str]) -> list[list[str]]:
    """
    Splits a list of file paths into a list of lists based on the 'Noise' value in the file path.
    """
    temp_list, final_list = [], []
    last_noise = None
    for fp in tqdm(json_paths, desc="Splitting JSON paths"):
        matches = re.findall(r'Noise = (\d+\.\d+)', fp)
        if not matches:
            continue  # Skip or handle files that don't match the pattern.
        noise = float(matches[0])
        if last_noise is None:
            last_noise = noise
        if noise != last_noise:
            final_list.append(temp_list)
            temp_list = []
            last_noise = noise
        temp_list.append(fp)
    final_list.append(temp_list)
    return final_list


def dump_json_paths(json_paths: list[str], output_file: str, verbose: bool = False) -> None:
    """
    Dumps a list of file paths to a text file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in json_paths:
            f.write(line + '\n')
    if verbose:
        print(f"Saved list of file paths to: `{output_file}`")


def dump_json_paths_wrapper(args: tuple[list[str], str, bool]) -> str:
    """
    Wrapper function for dumping JSON paths in parallel.
    """
    json_list, output_file, verbose = args
    dump_json_paths(json_list, output_file, verbose)
    return output_file


def dump_all_json_paths(
        list_of_json_lists: list[list[str]],
        verbose: bool = False,
        output_folder: str = "json_paths"
    ) -> bool:
    """
    Dumps each list of file paths from list_of_json_lists into a separate text file.
    """
    tasks = []
    for json_list in list_of_json_lists:
        if not json_list:
            continue
        fp = json_list[0]
        matches = re.findall(r'Network \(0*(\d+)\)', fp)
        if not matches:
            continue  # Skip or handle files that don't match the pattern.
        network_seed = int(matches[0])
        output_file = fp.replace('/resultados/', f'/{output_folder}/')
        output_file_parts = output_file.split('/')[:-1]
        output_file_parts[-1] = f'network_{network_seed:03d}.txt'
        output_file = '/'.join(output_file_parts)
        tasks.append((json_list, output_file, verbose))
    with ProcessPoolExecutor() as executor:
        list(tqdm(
            executor.map(dump_json_paths_wrapper, tasks),
            total=len(tasks),
            desc="Dumping JSON paths"
        ))
    return True


def dump_csv(txt_path: str, ignore_partition_key=True, output_folder: str = "csv_results") -> bool:
    """
    Reads a text file of file paths and dumps the JSON data into a CSV file.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        file_paths = [fp.strip() for fp in f.readlines()]
    data = []
    for fp in file_paths:
        with open(fp, 'r', encoding='utf-8') as f:
            file_data = json.load(f)  # Use json.load for file objects.
            for d in file_data:
                if ignore_partition_key and 'partition' in d:
                    del d['partition']
                data.append(d)
    df = pd.DataFrame(data)
    csv_path = txt_path.replace('/json_paths/', f'/{output_folder}/').replace('.txt', '.csv.gzip')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, compression="gzip")
    return True


def dump_csv_helper(args: tuple[str, str]) -> bool:
    """
    Dumps a CSV file from a text file of file paths.
    """
    txt_path, output_folder = args
    return dump_csv(txt_path, output_folder=output_folder)


def dump_all_csv(json_path: str, output_folder: str = "csv_results") -> bool:
    """
    Dumps all CSV files found in a directory tree.
    """
    sorted_txt_paths = get_paths_sorted(json_path, extension=".txt")
    args_list = [(fp, output_folder) for fp in sorted_txt_paths]
    with ProcessPoolExecutor() as executor:
        list(tqdm(
            executor.map(dump_csv_helper, args_list),
            total=len(sorted_txt_paths),
            desc="Dumping CSV files"
        ))
    return True


def read_csv_helper(fp: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a DataFrame.
    """
    return pd.read_csv(fp, compression="gzip")


def get_combined_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Gets a combined DataFrame from a list of CSV files.
    """
    sorted_csv_paths = get_paths_sorted(csv_path, extension=".csv.gzip")
    with ProcessPoolExecutor() as executor:
        df_list = list(tqdm(
            executor.map(read_csv_helper, sorted_csv_paths),
            total=len(sorted_csv_paths),
            desc="Combining CSV files"
        ))
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def load_experiment_data(results_folder: str):
    """
    Loads the experiment data from a given folder.
    """
    sorted_json_paths = get_paths_sorted(results_folder)
    split_paths = split_json_paths(sorted_json_paths)
    dump_all_json_paths(split_paths)
    time.sleep(2)
    json_path = results_folder.replace('/resultados/', '/json_paths/')
    dump_all_csv(json_path)
    time.sleep(2)
    csv_path = results_folder.replace('/resultados/', '/csv_results/')
    df = get_combined_dataframe(csv_path)
    return df


def main():
    """
    Main function to load the experiment data and save it to a CSV file.
    """
    results_folder = "/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados/"
    print("Loading experiment data from", results_folder)
    df = load_experiment_data(results_folder)
    output_path = results_folder[:-1] + '.csv.gzip'
    print('Saving data to', output_path)
    df.to_csv(output_path, index=False, compression="gzip")
    print("Done.")
    return True


if __name__ == '__main__':
    main()
