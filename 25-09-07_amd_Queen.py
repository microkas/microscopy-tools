# Lukas Nutz
# 22.07.2025

import os, re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from skimage import filters
from skimage.filters import threshold_li, threshold_otsu, threshold_mean
from skimage.morphology import opening, disk
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.measure import shannon_entropy
from scipy import stats
from imageio.v3 import imread, imwrite
import pyclesperanto_prototype as cle
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def main():
    run_acquisition(
        input_folder = "data", 
        ending_488="488.tif",
        ending_405="405.tif",
        output_folder = "results",
        target_size=(2048, 2048),
        gs_average = 3,                 
        th_radius_average = 50      
        )


def analyze_cells(images, output_folder):
    results = []
    for cell in regionprops(images["mask"]):
        cell_id = cell.label
        cell_mask = np.zeros_like(images["mask"], dtype=bool)
        cell_mask[tuple(cell.coords.T)] = True
        threshold_average = images["threshold_average"]    
        segment_average = (images["filtered_average"] > threshold_average) & cell_mask
        values_488 = images["img_488"][segment_average]
        values_405 = images["img_405"][segment_average]
        values_ratio = values_405/values_488
        perim = cell.perimeter
        area = cell_mask.sum()
        circularity = 4 * np.pi * area / (perim ** 2 + 1e-6)
        if len(values_405) > 1 and len(values_488) > 1:
            pcc, pval = stats.pearsonr(values_405, values_488)
        else:
            pcc, pval = np.nan, np.nan

        results.append({
            "well": images["well"], "pos": images["pos"], "time": images["time"],
            "cell_id": cell_id, "roi_area": area,
            "segmented_area_405": segment_average.sum(),   
            "mean_ratio": values_ratio.mean() if values_ratio.size else 0,          
            "std_ratio": np.std(values_ratio) if values_ratio.size else 0,                           
            "mean_405": values_405.mean() if values_405.size else 0,            
            "mean_488": values_488.mean() if values_488.size else 0,
            "sum_405": values_405.sum() if values_405.size else 0,
            "sum_488": values_488.sum() if values_488.size else 0,
            "circularity": circularity,
            "eccentricity": cell.eccentricity,
            "solidity": cell.solidity,
            "extent": cell.extent,
            "pcc": pcc,
            "p-val": pval,
            "rel_mito_488": values_488.sum()/(values_488.sum()+values_405.sum()) if values_488.size else 0 
        })

    df = pd.DataFrame(results)
    return results


def load_and_preprocess(mask_name, input_folder, output_folder, target_size, ending_488, ending_405,
                        gs_average=1, th_radius_average=50, entropy_threshold=5.0):
    debug_folder = os.path.join(output_folder, "debug_thresholds")
    os.makedirs(debug_folder, exist_ok=True)

    pattern = re.compile(
        r"^(?P<well>[^-]+)--W(?P<wellnum>\d+)--P(?P<pos>\d+)--Z\d+--T(?P<time>\d+)--(?P<slice>\d+|Trans \(CSU\))",
        re.IGNORECASE
    )
    m = pattern.search(mask_name)
    if not m:
        return None
    well, pos, t = m.group("well"), int(m.group("pos")), int(m.group("time"))

    loaded = load_images(mask_name, ending_488, ending_405, target_size, input_folder)
    if loaded is None:
        return None
    img_mask, img_488, img_405 = loaded

    se405 = shannon_entropy(img_405)
    se488 = shannon_entropy(img_488)
    if se405 < entropy_threshold or se488 < entropy_threshold:
        with open(os.path.join(output_folder, "skipped_low_entropy.txt"), "a") as f:
            f.write(f"{mask_name}, entropy_405={se405:.2f}, entropy_488={se488:.2f}\n")
        return None

    # average channel and filter
    img_average = (img_405 + img_488) / 2.0
    filtered_average = white_tophat(filters.gaussian(img_average, gs_average), th_radius_average) \
                       if th_radius_average != 0 else filters.gaussian(img_average, gs_average)

    threshold_average = threshold_otsu(filtered_average)
    segment_average_mask = filtered_average > threshold_average

    # debug dumps (rename to 'avg' to avoid confusion)
    #imwrite(os.path.join(debug_folder, f"{mask_name}_avg_filtered.png"), filtered_average)
    #imwrite(os.path.join(debug_folder, f"{mask_name}_avg_seg.png"), segment_average_mask)

    # Return keys that analyze_cells expects (map avg -> "405" keys)
    return {
        "mask_name": mask_name,
        "well": well,
        "pos": pos,
        "time": t,
        "mask": img_mask,
        "img_405": img_405,
        "img_488": img_488,
        "filtered_average": filtered_average,     # <-- mapped for compatibility
        "threshold_average": threshold_average    # <-- mapped for compatibility
    }


    
def load_images(mask_name, ending_488, ending_405, target_size, input_folder):
    path_mask = os.path.join(input_folder, mask_name)
    if not os.path.exists(path_mask):
        return None

    # strip the suffix "Trans (CSU)_cp_masks.png" and append the provided endings
    base = mask_name.replace("Trans (CSU)_cp_masks.png", "")
    path_488 = os.path.join(input_folder, f"{base}{ending_488}")
    path_405 = os.path.join(input_folder, f"{base}{ending_405}")

    if not (os.path.exists(path_488) and os.path.exists(path_405)):
        return None

    # mask -> label and resize with nearest (order=0)
    mask = imread(path_mask).astype(int)
    img_mask = resize(label(mask), target_size, order=0, preserve_range=True, anti_aliasing=False)

    # channels -> float32 and resize to target_size (bilinear)
    img_488 = imread(path_488).astype(np.float32)
    img_405 = imread(path_405).astype(np.float32)

    if img_488.shape != target_size:
        img_488 = resize(img_488, target_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    if img_405.shape != target_size:
        img_405 = resize(img_405, target_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    return img_mask, img_488, img_405

def white_tophat(img, radius):
    img_gpu = cle.push(img.astype(np.float32))
    eroded = cle.minimum_sphere(img_gpu, None, radius_x=radius, radius_y=radius)
    opened = cle.maximum_sphere(eroded, None, radius_x=radius, radius_y=radius)
    wth = cle.subtract_images(img_gpu, opened)
    return cle.pull(wth)


def is_low_entropy(img, threshold=5.0):
    return shannon_entropy(img) < threshold


def to_uint8(img):
    return rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)


def run_acquisition(input_folder, ending_488, ending_405, output_folder, target_size, 
                    gs_average, th_radius_average, max_workers=4, max_in_flight=8):
    os.makedirs(output_folder, exist_ok=True)
    masks = [f for f in os.listdir(input_folder) if f.endswith("_cp_masks.png")]
    results = []
    def loader():
        for mask in masks:
            images = load_and_preprocess(mask, input_folder, output_folder, target_size, 
                                         ending_488, ending_405, gs_average, th_radius_average,)
            if images:
                yield images
                
    print(f"Starting analysis of {len(masks)} files...")
    with ProcessPoolExecutor(max_workers) as executor:
        futures = set()
        images_iter = loader()
        for _ in range(max_in_flight):
            try:
                futures.add(executor.submit(analyze_cells, next(images_iter), output_folder))
            except StopIteration:
                break
        completed = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.remove(fut)
                try:
                    res = fut.result()
                    if res: results.extend(res)
                except Exception as e:
                    print(f"⚠️ Task failed: {e}")
                completed += 1
                print(f"\rProcessed {completed}/{len(masks)}", end="")
                try:
                    futures.add(executor.submit(analyze_cells, next(images_iter), output_folder))
                except StopIteration:
                    pass
                
    if results:
        cell_data = pd.DataFrame(results)
        cell_data.to_excel(os.path.join(output_folder, "all_cells.xlsx"), index=False)
        print("\nSummary saved.")
    else:
        print("\nNo results found.")
        return None
        
    return cell_data 


if __name__ == "__main__":       
    main()                       



