import os
import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="the path of input folder file")
parser.add_argument("-out", "--path_output", type=str, help="the path of output folder file")
parser.add_argument("-script", "--path_script", type=str, help="the path of .py script")
parser.add_argument("-num", "--num_threads", type=int, help="the number of threads")
parser.add_argument("-old_ext", "--old_filename_extension", type=str, help="the extension of the files to be processed")
parser.add_argument("-new_ext", "--new_filename_extension", type=str, help="the new extension of the files to be processed")
parser.add_argument("-w", "--window_size", type=int, help="the window_size of series feature")

def process_file(file, output_folder, script_path, old_filename_extension, new_filename_extension, window_size):
    output_file = os.path.join(output_folder, os.path.basename(file).replace(old_filename_extension, new_filename_extension))
    subprocess.run(["python", script_path, "-in", file, "-out", output_file, "-w", str(window_size)])

def process_files_in_parallel(input_folder, output_folder, script_path, num_threads, old_filename_extension, new_filename_extension, window_size):
    files = glob.glob(os.path.join(input_folder, f"*{old_filename_extension}"))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(lambda file: process_file(file, output_folder, script_path, old_filename_extension, new_filename_extension, window_size), files), total=len(files)))

if __name__ == "__main__":
    args = parser.parse_args()
    input_folder = args.path_input
    output_folder = args.path_output
    script_path = args.path_script
    num_threads = args.num_threads
    old_filename_extension = args.old_filename_extension
    new_filename_extension = args.new_filename_extension
    window_size = args.window_size

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_files_in_parallel(input_folder, output_folder, script_path, num_threads, old_filename_extension, new_filename_extension, window_size)
