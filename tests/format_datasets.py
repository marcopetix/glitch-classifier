import os
import shutil
import tarfile
import zipfile
import argparse

def extract_png_dataset(png_tar_path, output_dir):
    output_dir = os.path.join(output_dir, "png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Extracting PNG dataset from {png_tar_path} to {output_dir}")
    
    with tarfile.open(png_tar_path, "r:gz") as tar:
        print(f"Extracting {png_tar_path}...")
        tar.extractall(path=output_dir)
        print("Extraction complete.")


def extract_npy_dataset(zip_path, output_dir):
    output_dir = os.path.join(output_dir, "npy")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting NPZ dataset from {zip_path} to {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    for fname in os.listdir(output_dir):
        if fname.endswith(".tar.gz"):
            fpath = os.path.join(output_dir, fname)
            try:
                with tarfile.open(fpath, "r:gz") as tar:
                    tar.extractall(path=output_dir)
            except tarfile.ReadError:
                try:
                    with tarfile.open(fpath, "r:") as tar:
                        tar.extractall(path=output_dir)
                except Exception as e:
                    print(f"Failed to extract {fpath}: {e}")





def main():
    parser = argparse.ArgumentParser(description="Extract and restructure datasets.")
    parser.add_argument("--dataset-type", type=str, choices=["png", "npy"], required=True,
                        help="Type of dataset to extract: 'png' or 'npy'")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to the dataset archive (e.g., .tar.gz or .zip file)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Directory where the structured dataset will be saved")
    args = parser.parse_args()

    if args.dataset_type == "png":
        extract_png_dataset(args.input_path, args.output_dir)
    elif args.dataset_type == "npy":
        extract_npy_dataset(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()