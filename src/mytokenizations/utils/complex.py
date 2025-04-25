import os
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
import numpy as np
import shutil


def check_existed_files(dir1: Union[str, Path], dir2: Union[str, Path]):
    """Check which files (just names not extensions) from path1, exists in path2.
    Quite brute-force implementation. There should be a more efficient way."""

    dir1 = Path(dir1)
    dir2 = Path(dir2)

    dir1_names = []
    dir2_names = []

    only_exists_in_1 = []
    exists_both = []
    only_exists_in_2 = []

    dir1_files = [f for f in os.listdir(dir1) if os.path.isfile(f)]
    dir2_files = [f for f in os.listdir(dir2) if os.path.isfile(f)]

    for file1 in dir1_files:
        dir1_names.append(Path(file1).name)

    for file2 in dir2_files:
        dir2_names.append(Path(file2).name)

    for elem in dir1_names:
        if elem not in dir2_names:
            only_exists_in_1.append(elem)
            continue
        elif elem in dir2_names:
            exists_both.append(elem)
            continue
        else:
            pass

    for elem in dir2_names:
        if elem not in exists_both:
            only_exists_in_2.append(elem)

    return only_exists_in_1, exists_both, only_exists_in_2


def move_files(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    start_str: str,
    end_str: str = "json",
):
    """Moves the files from input_paths that start wih 'start_str'.

    Args:
        input_path (Union[str, Path]):
        output_dir (Union[str, Path]):
        start_str (str):
        end_str (str, optional): Defaults to "json".
    """
    total_moved = 0
    input_paths = list(Path(input_path).glob(f"*.{end_str}"))
    for path in tqdm(input_paths):
        if str(path.name)[: len(start_str)] == start_str:
            shutil.move(path, str(Path(output_dir) / Path(path.name)))
            total_moved += 1
    lg.info(f"Total files moved: {total_moved}")


def json2pickle(input_path: Union[str, Path], output_path: Union[str, Path]):
    """Converts a .json file to .pickle file. Give pathes to the full name including the extensions."""

    lg.info("Loading the data.")
    with open(input_path, "r") as d:
        x = json.load(d)

    lg.info("Saving the data.")
    out_pickle(x, output_path, True)


def merge_files(input_path: Union[str, Path], file_type: str, onthefly):
    """Merges the :file_type: files into one. :file_type: can be either json or pickle.
    Outputs the merged file into the directory as f'merged.{file_type}'
    input_path is dir. Also, via an `onthefly` function, one may do certain
    processes just after importing the data."""

    file_type = "p" if file_type == "pickle" else file_type
    paths = Path(input_path).glob(f"*.{file_type}")

    all_data = []

    for file_path in paths:
        if file_type == "json":
            all_data.append(onthefly(in_json(file_path)))
        elif file_type == "p":
            all_data.append(onthefly(in_pickle(file_path)))
        else:
            raise Exception("file_type param should be 'json' or 'pickle'.")

    output_path = str(input_path) + f"/merged.{file_type}"
    if file_type == "json":
        with open(output_path, "w", encoding="utf-8") as d:
            json.dump(all_data, d)
    elif file_type == "p":
        out_pickle(all_data, output_path, True)
    else:
        raise Exception("file_type param should be 'json' or 'pickle'.")


def merge_lists(data: List[List[Any]], merge_num: int):
    """Concat merge_num many lists into a new list."""

    data = data[: len(data) - (len(data) % merge_num)]
    final = []
    for i in range(0, len(data), merge_num):
        _tmp = []
        for j in range(i, i + merge_num):
            _tmp += data[j]
        final.append(_tmp)

    return final


def map_to_range(
    numbers: List[int],
    old_min: int = 4,
    old_max: int = 131,
    new_min: int = 1,
    new_max: int = 10,
) -> List[int]:
    """Maps a list of integers into one scale to another.

    Args:
        numbers (List[int]): list of integers.
        old_min (int, optional): Defaults to 4.
        old_max (int, optional): Defaults to 131.
        new_min (int, optional): Defaults to 1.
        new_max (int, optional): Defaults to 10.

    Returns:
        List[int]: mapped list
    """
    return np.round(
        ((numbers - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    ).astype(int)
