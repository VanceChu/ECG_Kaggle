import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def resample_pixels_in_dir(dir, resample_factor):
    error_list = []
    for root, _, files in os.walk(dir):
        print(f"Running increased pixel density on {root}...")
        for file in tqdm(files):
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r") as file:
                        data = json.load(file)

                    leads = data["leads"]
                    for i in range(len(leads)):
                        pixels = np.array(leads[i]["plotted_pixels"])
                        # Use linear interpolation to add more pixels to the plot
                        new_pixels = np.zeros(((len(pixels) - 1) * resample_factor, 2))
                        for j in range(len(pixels) - 1):
                            new_pixels[
                                j * resample_factor : (j + 1) * resample_factor, 0
                            ] = np.linspace(
                                pixels[j, 0], pixels[j + 1, 0], resample_factor
                            )
                            new_pixels[
                                j * resample_factor : (j + 1) * resample_factor, 1
                            ] = np.linspace(
                                pixels[j, 1], pixels[j + 1, 1], resample_factor
                            )
                        data["leads"][i]["dense_plotted_pixels"] = new_pixels.tolist()

                    if "leads_augmented" in data.keys():
                        leads = data["leads_augmented"]
                        for i in range(len(leads)):
                            pixels = np.array(leads[i]["plotted_pixels"])
                            # Use linear interpolation to add more pixels to the plot
                            new_pixels = np.zeros(
                                ((len(pixels) - 1) * resample_factor, 2)
                            )
                            for j in range(len(pixels) - 1):
                                new_pixels[
                                    j * resample_factor : (j + 1) * resample_factor, 0
                                ] = np.linspace(
                                    pixels[j, 0], pixels[j + 1, 0], resample_factor
                                )
                                new_pixels[
                                    j * resample_factor : (j + 1) * resample_factor, 1
                                ] = np.linspace(
                                    pixels[j, 1], pixels[j + 1, 1], resample_factor
                                )
                            data["leads_augmented"][i]["dense_plotted_pixels"] = (
                                new_pixels.tolist()
                            )

                    with open(file_path, "w") as file:
                        json.dump(data, file, indent=4)
                except Exception as e:
                    error_list.append((e, file_path))
    print("Errors:")
    print(error_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample plotted pixels in a " "directory."
    )

    parser.add_argument(
        "--resample_factor",
        type=int,
        default=20,
        help="Multiplicative factor for resampling the plotted pixels.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="ptb-xl/records500_prepared_w_images",
        help="Directory containing plotted pixels to be resampled.",
    )
    parser.add_argument(
        "--run_on_subdirs",
        action="store_true",
        help="Whether to run on folder itself or in parallel on subdirs?.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Whether to plot the resampled pixels."
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")

    args = parser.parse_args()

    # Increase pixel density
    dir = args.dir
    if args.run_on_subdirs:
        dirs = [
            os.path.join(dir, d)
            for d in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, d))
        ]
        print(
            f"Running in parallel using {args.num_workers}/{os.cpu_count()} workers for {len(dirs)} dirs: {dirs}"
        )
        resample_pixels_in_dir_partial = partial(
            resample_pixels_in_dir, resample_factor=args.resample_factor
        )
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(
                tqdm(executor.map(resample_pixels_in_dir_partial, dirs), total=len(dirs))
            )
    else:
        print("Running on all files in dir and subdirs...")
        resample_pixels_in_dir(dir, args.resample_factor)
    print("All files saved.")

    # Plot
    if args.plot:
        for file in os.listdir(dir):
            if file.endswith(".json"):
                file_path = os.path.join(dir, file)

                with open(file_path, "r") as file:
                    data = json.load(file)

                leads = data["leads"]
                for i in range(len(leads)):
                    pixels = np.array(leads[i]["plotted_pixels"])
                    plt.scatter(pixels[:, 1], -pixels[:, 0], s=1)
                plt.figure()
                for i in range(len(leads)):
                    dense_pixels = np.array(leads[i]["dense_plotted_pixels"])
                    plt.scatter(dense_pixels[:, 1], -dense_pixels[:, 0], s=1)
                plt.show()
                break

    print("Done with replot_pixels.py")
