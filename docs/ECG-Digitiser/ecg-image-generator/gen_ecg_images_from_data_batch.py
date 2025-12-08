import os, sys, argparse
import random
import gc
import csv
from helper_functions import find_records
from gen_ecg_image_from_data import run_single_file
import warnings
from tqdm import tqdm

import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-se', '--seed', type=int, required=False, default = -1)
    parser.add_argument('--num_leads',type=str,default='twelve')
    parser.add_argument('--max_num_images',type=int,default = -1)
    parser.add_argument('--config_file', type=str, default='config.yaml')
    
    parser.add_argument('-r','--resolution',type=int,required=False,default = 200)
    parser.add_argument('--pad_inches',type=int,required=False,default=0)
    parser.add_argument('-ph','--print_header', action="store_true",default=False)
    parser.add_argument('--num_columns',type=int,default = -1)
    parser.add_argument('--full_mode', type=str,default='II')
    parser.add_argument('--mask_unplotted_samples', action="store_true", default=False)
    parser.add_argument('--add_qr_code', action="store_true", default=False)

    parser.add_argument('-l', '--link', type=str, required=False,default='')
    parser.add_argument('-n','--num_words',type=int,required=False,default=5)
    parser.add_argument('--x_offset',dest='x_offset',type=int,default = 30)
    parser.add_argument('--y_offset',dest='y_offset',type=int,default = 30)
    parser.add_argument('--hws',dest='handwriting_size_factor',type=float,default = 0.2)
    
    parser.add_argument('-ca','--crease_angle',type=int,default=90)
    parser.add_argument('-nv','--num_creases_vertically',type=int,default=10)
    parser.add_argument('-nh','--num_creases_horizontally',type=int,default=10)

    parser.add_argument('-rot','--rotate',type=int,default=0)
    parser.add_argument('-noise','--noise',type=int,default=50)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=40000)

    parser.add_argument('--random_resolution',action="store_true",default=False)
    parser.add_argument('--random_padding',action="store_true",default=False)
    parser.add_argument('--random_grid_color',action="store_true",default=False)
    parser.add_argument('--standard_grid_color', type=int, default=5)
    parser.add_argument('--calibration_pulse',type=float,default=1)
    parser.add_argument('--random_grid_present',type=float,default=1)
    parser.add_argument('--random_print_header',type=float,default=0)
    parser.add_argument('--random_bw',type=float,default=0)
    parser.add_argument('--remove_lead_names',action="store_false",default=True)
    parser.add_argument('--lead_name_bbox',action="store_true",default=False)
    parser.add_argument('--store_config', type=int, nargs='?', const=1, default=0)

    parser.add_argument('--deterministic_offset',action="store_true",default=False)
    parser.add_argument('--deterministic_num_words',action="store_true",default=False)
    parser.add_argument('--deterministic_hw_size',action="store_true",default=False)

    parser.add_argument('--deterministic_angle',action="store_true",default=False)
    parser.add_argument('--deterministic_vertical',action="store_true",default=False)
    parser.add_argument('--deterministic_horizontal',action="store_true",default=False)

    parser.add_argument('--deterministic_rot',action="store_true",default=False)
    parser.add_argument('--deterministic_noise',action="store_true",default=False)
    parser.add_argument('--deterministic_crop',action="store_true",default=False)
    parser.add_argument('--deterministic_temp',action="store_true",default=False)

    parser.add_argument('--fully_random',action='store_true',default=False)
    parser.add_argument('--hw_text',action='store_true',default=False)
    parser.add_argument('--wrinkles',action='store_true',default=False)
    parser.add_argument('--augment',action='store_true',default=False)
    parser.add_argument('--lead_bbox',action='store_true',default=False)
    
    parser.add_argument('--num_images_per_ecg',type=int,default=None)
    parser.add_argument('--run_in_parallel',action='store_true',default=False)
    parser.add_argument('--num_workers',type=int,default=-1)
    parser.add_argument('--overwrite',action='store_true',default=False)

    return parser


def find_file_in_directory(filename, directory):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            relative_path = os.path.relpath(root, directory)
            if relative_path == ".":
                return None
            else:
                return os.path.join(relative_path)
    return None


def prepare_args(args, full_header_file, full_recording_file, original_output_dir):
    """For parallel processing, we need to create a new args object for each file.
    
    This helps to avoid race conditions and other issues that may arise when multiple processes
    """
    
    modified_args = copy.deepcopy(args)
    filename = full_recording_file
    header = full_header_file
    folder_struct_list = full_header_file.split('/')[:-1]

    modified_args.input_file = os.path.join(args.input_directory, filename)
    modified_args.header_file = os.path.join(args.input_directory, header)
    modified_args.start_index = -1
    modified_args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
    modified_args.encoding = os.path.split(os.path.splitext(filename)[0])[1]
    
    return modified_args


def execute_parallel(prepared_args, num_workers):
    if num_workers == -1:
        workers = os.cpu_count() - 2
    else:
        workers = num_workers
    print(f"Using {workers}/{os.cpu_count()} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        _ = list(tqdm(executor.map(run_single_file, prepared_args), total=len(prepared_args)))
    executor.shutdown(wait=True)
    gc.collect()


def run(args):
        random.seed(args.seed)

        if os.path.isabs(args.input_directory) == False:
            args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
        if os.path.isabs(args.output_directory) == False:
            original_output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
        else:
            original_output_dir = args.output_directory
        
        if os.path.exists(args.input_directory) == False or os.path.isdir(args.input_directory) == False:
            raise Exception("The input directory does not exist, Please re-check the input arguments!")

        if os.path.exists(original_output_dir) == False:
            os.makedirs(original_output_dir)

        i = 0
        full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir)
        
        if args.max_num_images != -1:
            print(f"Reducing the number of files to {args.max_num_images}")
            full_header_files = full_header_files[:args.max_num_images]
            full_recording_files = full_recording_files[:args.max_num_images]
        assert len(full_header_files) == len(full_recording_files), "Number of header files and recording files do not match"
        
        # Prepare arguments for each file pair
        prepared_args = [prepare_args(args, full_header_file, full_recording_file, original_output_dir) for full_header_file, full_recording_file in zip(full_header_files, full_recording_files)]
        
        if args.overwrite:
            print("Overwriting existing files.")
        
        # Run the files in parallel or sequentially
        if args.run_in_parallel:
            print("Running in parallel...")
            execute_parallel(prepared_args, args.num_workers)
        else:
            print("Running sequentially...")
            for a in tqdm(prepared_args):
                # Only run if header_file plus one does not yet exists in the output directory
                if not args.overwrite:
                    header_file_name = os.path.split(a.header_file)[1]
                    header_file_index_to_check = str(int(header_file_name.split('_')[0]) + 1).zfill(5)
                    header_file_name_to_check = header_file_index_to_check + '_' +  header_file_name.split('_')[1]
                    sub_dir = find_file_in_directory(header_file_name_to_check, args.input_directory)
                    if sub_dir is not None:
                        header_file_to_check = os.path.join(args.output_directory, sub_dir, header_file_name_to_check)
                    else:
                        header_file_to_check = os.path.join(args.output_directory, header_file_name_to_check)
                    if not os.path.exists(header_file_to_check):
                        run_single_file(a)
                else:
                    run_single_file(a)
                gc.collect()
        print("Done!")

if __name__=='__main__':
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    run(get_parser().parse_args(sys.argv[1:]))
