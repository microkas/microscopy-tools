# 15.09.2025
# Lukas Nutz

import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tifffile import imread
from skimage import exposure, img_as_ubyte, transform, filters
from imageio import imsave
from matplotlib import pyplot as plt

input_folder = "data"
output_folder = os.path.join(input_folder, "png")
os.makedirs(output_folder, exist_ok=True)

pattern = re.compile(r"P(\d+)--Z\d+--T(\d+)--(\d+|Trans \(CSU\))", re.IGNORECASE)
wavelength_to_channel = {'Trans (CSU)': 0, '488': 1, '406': 2}
channels_for_cellpose = 0
target_size = (512, 512)

def dog_laplace_approx(filename, sigma1=2, sigma2=3):
    match = pattern.search(filename)
    if not match:
        print(f"⚠️ Skipping unmatched file: {filename}")
        return None

    wl = match.group(3)
    if wl != "Trans (CSU)":
        return None 

    channel_idx = wavelength_to_channel[wl]
    fpath = os.path.join(input_folder, filename)
    try:
        img = imread(fpath)
    except Exception as e:
        print(f"❌ Could not read {filename}: {e}")
        return None

    if img.ndim == 3:
        try:
            img = img[channel_idx]
        except IndexError:
            print(f"❌ Channel index {channel_idx} out of range for {filename}")
            return None


    # Difference of Gaussians and scale to target size
    dog = filters.gaussian(img, sigma=sigma1) - filters.gaussian(img, sigma=sigma2)
    dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
    dog_u8 = img_as_ubyte(dog_norm)
    dog_resized = transform.resize(dog_u8, target_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)

    # Save as PNG
    outname = filename.rsplit(".", 1)[0] + ".png"
    outpath = os.path.join(output_folder, outname)
    imsave(outpath, dog_resized)

    return outpath


if __name__ == "__main__":
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
    results = []

    print(f"Processing {len(files)} files...")

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(dog_laplace_approx, f): f for f in files}
        completed = 0

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                fname = futures.pop(fut)
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                    else:
                        raise ValueError("Returned None")
                except Exception as e:
                    print(f"\n🔁 Failed {fname}: {e}")
                completed += 1
                print(f"\rProcessed {completed}/{len(files)} files", end="")

    print("\n✅ Finished.")

