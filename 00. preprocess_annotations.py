import pandas as pd
import os
from tqdm import tqdm
import warnings
import numpy as np
import torch
from ultralytics.utils.metrics import mask_iou
import itertools
import matplotlib.image as mpimg

# Ignore warnings
warnings.filterwarnings('ignore')

def create_separate_masks(annoations, class_ids, height, width):
    masks = []

    for segment, (pixel_str, class_id) in enumerate(zip(annoations, class_ids)):
        mask = np.zeros((height, width)).reshape(-1)
        splitted_pixels = list(map(int, pixel_str.split()))
        pixel_starts = splitted_pixels[::2]
        run_lengths = splitted_pixels[1::2]
        assert max(pixel_starts) < mask.shape[0]
        for pixel_start, run_length in zip(pixel_starts, run_lengths):
            pixel_start = int(pixel_start) - 1
            run_length = int(run_length)
            mask[pixel_start:pixel_start+run_length] = 1
        masks.append(mask.reshape((height, width), order='F'))
    return masks

def flatten_mask(mask):

    flattened_mask = mask.flatten()
    mask_tensor = np.reshape(flattened_mask, (1, -1))

    mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)
    return mask_tensor

def check_overlap(mask1, mask2, threshold=0.9):
    """
    Determine if the overlap between two masks covers more than `threshold` of the smaller mask.
    """
    # Calculate IoU using the mask_iou function
    iou = mask_iou(mask1, mask2).item()
    if iou==0:
        return False
    
    # Calculate the areas of the masks
    area1 = mask1.sum().item()
    area2 = mask2.sum().item()
    
    # Determine the smaller mask
    smaller_area = min(area1, area2)
    
    # Calculate the intersection area based on IoU and union
    intersection = iou * (area1 + area2) / (1 + iou)
    
    # Check if the intersection covers more than threshold of the smaller mask
    if intersection / smaller_area > threshold:
        return True
    else:
        return False
    
def search_attribute_pairs(tmp_df, image_base_path='imaterialist-fashion-2020-fgvc7/train'):
    tmp = tmp_df.reset_index(drop=True).copy()
    image = mpimg.imread(os.path.join(image_base_path, tmp.ImageId.unique()[0]+'.jpg'))
    # Create binary masks
    masks = create_separate_masks(tmp['EncodedPixels'], tmp['ClassId'], tmp['Height'].values[0], tmp['Width'].values[0])

    combinations = list(itertools.combinations(range(len(masks)), 2))

    pairs = list()

    # Consider all combinations
    for comb in combinations:
        # Select binary masks
        mask1 = masks[comb[0]]
        mask2 = masks[comb[1]]
        # Flatten for comparison
        flat1 = flatten_mask(mask1)
        flat2 = flatten_mask(mask2)
        # Check if the smaller mask shares more than 90% of pixels with the larger mask
        if check_overlap(flat1, flat2):
            # Select the larger mask as the representative mask
            if mask1.sum() > mask2.sum():
                pairs.append([comb[0], comb])
            else:
                pairs.append([comb[1], comb])
    return pairs

def merge_attribute_pairs(tmp_df, pairs):
    tmp = tmp_df.reset_index(drop=True).copy()
    # Convert to string value since some cases may not have attributes
    tmp.loc[tmp['AttributesIds'].isna(), 'AttributesIds'] = ''
    main_pairs = list(set([i[0] for i in pairs]))

    for mp in main_pairs:
        # Select pairs containing the main category
        pair = [i[1] for i in pairs if i[0]==mp]
        # Select only other IDs excluding the main category == sub-categories
        flat_pair = list(set([element for tuple_ in pair for element in tuple_]))
        sub_category = [i for i in flat_pair if i!=mp]
        # Merge attributes of sub-categories into one
        sub_attributes = tmp.loc[sub_category, 'AttributesIds'].values
        sub_attributes = list(set(','.join(sub_attributes).split(',')))
        sub_attributes = ','.join(sub_attributes)
        # Save as second attribute of the main category
        tmp.loc[mp, 'second_AttributesIds'] = sub_attributes

    return tmp

### functions for batch processing
def divide_into_batches(dataframe, batch_size):
    """Yield successive n-sized chunks from dataframe."""
    images = dataframe['ImageId'].unique()
    batches = [images[i:i+3] for i in range(0, len(images), batch_size)]

    for batch in batches:
        yield dataframe.loc[dataframe['ImageId'].isin(batch)]


def process_and_append_to_file(batch, image_base_path, output_file):
    processed_batch = [process_per_image((image, batch, image_base_path)) for image in batch['ImageId'].unique()]
    # Convert each DataFrame in processed_batch to a line-delimited JSON string
    with open(output_file, 'a') as f_out:
        for df in processed_batch:
            json_string = df.to_json(orient='records')
            f_out.write(json_string + '\n')  # Write each batch as a new line

def process_per_image(args):
    image, anno, image_base_path = args

    # Filter the DataFrame for the current image
    tmp_df = anno[anno['ImageId'] == image].copy()
    # Search for attribute pairs within this image's data
    pairs = search_attribute_pairs(tmp_df, image_base_path)
    # If pairs are found, merge their attributes
    if len(pairs) > 0:
        tmp_df = merge_attribute_pairs(tmp_df, pairs)
    return tmp_df

def parallel_process_images(anno, image_base_path, output_file, batch_size=20):
    batches = list(divide_into_batches(anno, batch_size))
    for batch in tqdm(batches, desc="Processing Batches"):
        process_and_append_to_file(batch, image_base_path, output_file)

if __name__ == "__main__":
    anno = pd.read_csv('imaterialist-fashion-2020-fgvc7/train.csv')

    # Specify the image base path if different from the default
    image_base_path = 'imaterialist-fashion-2020-fgvc7/train'

    output_file = "outputs2.json"

    open(output_file, 'w').close()
    print("Start process")
    parallel_process_images(anno, image_base_path, output_file, batch_size=5)

    # new_anno.to_csv("imaterialist-fashion-2020-fgvc7/new_annotations.csv", index=False)