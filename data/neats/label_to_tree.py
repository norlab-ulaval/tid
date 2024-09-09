#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:19:10 2024

@author: vincent
"""
import random
from collections import defaultdict
import os
from os import listdir
from os.path import join, splitext
import json
from tqdm import tqdm


def build_label_tree(image_folder):
  """
  Builds a label tree from image filenames with a specific format.

  Args:
    image_folder: Path to the folder containing images.

  Returns:
    A dictionary representing the label tree with family, genus, and species.
  """

  label_tree = defaultdict(lambda: defaultdict(set))  # Nested dictionary with sets for unique species

  # Loop through files in the folder
  for filename in tqdm(os.listdir(image_folder)):
    if not os.path.isfile(os.path.join(image_folder, filename)):  # Skip non-files
      continue

    # Extract information from filename
    parts = filename.split("_")
    
    observation_id, family, genus, species = parts[:4]
    
    if genus == "nan":
        # print(f"Warning: Skipping image {filename} - incorrect format")
        continue

    # Build the label tree structure
    label_tree[family][genus].add(species)

  return label_tree


# Example usage (replace 'path/to/images' with your actual folder path)
#folder_dir = "/gpfs/groups/gc069/dm9878/datasets/iNatTrees/images_observations_405362_2021-07-01_2022-01-01/"
folder_dir = "/gpfs/groups/gc069/datasets/neats/"
label_tree = build_label_tree(folder_dir)

# Count the total number of unique species from label_tree
total_species = 0
# Print the label tree structure
for family, genus_data in label_tree.items():
  print(f"Family: {family}")
  for genus, species in genus_data.items():
    print(f"\tGenus: {genus}")
    for specie in species:
      print(f"\t\tSpecies: {specie}")
      total_species += 1
      
print(f"Total unique species: {total_species}")



def split_dataset_by_occurrence(image_folder, label_tree, test_split=0.1, val_split=0.1, min_test_samples_per_species=100):
  """
  Splits a dataset of images with occurrence information into test, validation, and train sets.
  Ensures at least `min_test_samples_per_species` images per species (considering the label tree) in the test set,
  while keeping only single-occurrence (x=1) images in test and validation sets.

  Args:
    image_folder: Path to the folder containing images.
    label_tree: A dictionary representing the label hierarchy (provided by build_label_tree function).
    test_split: Proportion of images for the test set (default: 0.1).
    val_split: Proportion of images for the validation set (default: 0.1).
    min_test_samples_per_species: Minimum number of images per species in the test set (default: 1).

  Returns:
    A dictionary containing paths for each set: test, validation, and train.
  """

  # Initialize image sets
  image_sets = {"test": [], "validation": [], "train": []}
  seen_objects = defaultdict(int)  # Track object occurrences with default value 0
  train_species_counts = defaultdict(int)  # Track species count in test set, considering label tree
  test_species_counts = defaultdict(int)  # Track species count in test set, considering label tree
  val_species_counts = defaultdict(int)  # Track species count in val set, considering label tree

  # Process each image in the folder
  for filename in tqdm(listdir(image_folder)):
    if not filename.endswith((".jpg", ".png")):  # Check for image extensions
      continue

    # Extract information from filename (without extension)
    base_filename, _ = splitext(filename)
    parts = base_filename.split("_")
    if len(parts) < 5:
      # print(f"Warning: Skipping image {filename} - incorrect format")
      continue

    # Separate taxon
    observation_id, family, genus, species = parts[:4]
    taxon = family + "_" + genus + "_" + species

    if genus == "nan":
      # print(f"Warning: Skipping image {filename} - incorrect format")
      continue

    # Separate occurrence information (i_of_x)
    occurrence, _, total_occurrences = parts[4:]

    occurrence = int(occurrence)
    total_occurrences = int(total_occurrences)

    # Track object occurrences
    seen_objects[(family, genus, species)] += 1

    # Get label information from label tree (if provided)
    label = None
    if label_tree:
      current_node = label_tree
      for part in (family, genus, species):
        if isinstance(current_node, dict) and part in current_node:
          current_node = current_node[part]
        else:
          break
      else:
        label = current_node.get("label", None)  # Extract label if available

    # Select images based on occurrence
    if total_occurrences == 1:  # Only consider single-occurrence images (x=1)
      # Find all species under the current node (considering label tree)
      all_species_under_node = set()
      if label_tree and family in label_tree and genus in label_tree[family]:
        all_species_under_node = label_tree[family][genus]

      # Calculate minimum samples considering all child species
      min_samples_for_node = min(min_test_samples_per_species, len(all_species_under_node))

      # Check if enough samples in test set for this node (species or group)
      if (family, genus, species) not in image_sets["test"] or \
         test_species_counts[(family, genus, species)] < min_samples_for_node:

        # Randomly assign to test or validation set with adjusted probabilities
        test_prob = test_split
        val_prob = val_split

        if random.random() < test_prob:
          image_sets["test"].append(join(image_folder, filename))
          test_species_counts[(family, genus, species)] += 1  # Increment test count for this node
        elif random.random() < test_prob + val_prob:
          image_sets["validation"].append(join(image_folder, filename))
          val_species_counts[(family, genus, species)] += 1  # Increment val count for this node
        else:
          image_sets["train"].append(join(image_folder, filename))
          train_species_counts[(family, genus, species)] += 1  # Increment train count for this node
      else:
        # If enough samples in test set for this node, add to train set directly
        image_sets["train"].append(join(image_folder, filename))
        train_species_counts[(family, genus, species)] += 1  # Increment train count for this node
    else:
      image_sets["train"].append(join(image_folder, filename))
      train_species_counts[(family, genus, species)] += 1  # Increment train count for this node

  # Print summary information
  total_images = len(image_sets["test"]) + len(image_sets["validation"]) + len(image_sets["train"])
  print(f"Total images: {total_images}")
  print(f"Test set: {len(image_sets['test'])} images")  # Use exact number of images
  print(f"Validation set: {len(image_sets['validation'])} images")  # Use exact number of images
  print(f"Train set: {len(image_sets['train'])} images")  # Use exact number of images

  # Save results as JSON files
  # for name, image_list in image_sets.items():
  #   with open(f"{name}.json", 'w') as outfile:
  #     json.dump(image_list, outfile, indent=4)

  return image_sets, test_species_counts, val_species_counts, train_species_counts

if __name__ == "__main__":
  # Replace with your actual folder paths
  image_folder = folder_dir
  # output_folder = "path/to/output/folder"

  image_sets, test_species_counts, val_species_counts, train_species_counts = split_dataset_by_occurrence(image_folder, label_tree, test_split=0.1, val_split=0.1)

  print(len(test_species_counts))
  print(len(val_species_counts))
  print(len(train_species_counts))


def create_coco_annotations(image_list, folder_dir, output_filename, mode='train'):
    """
    Creates a COCO annotation JSON file from a folder of images.
    
    Args:
      image_folder: Path to the folder containing images.
      output_filename: Name of the output JSON file.
    """
    
    # Initialize COCO structures
    info = {
        "year": 2024,  
        "version": "1.0",
        "description": "COCO annotations for NEATS dataset",
        "contributor": "Vincent Grondin",  
    }
    images = []
    categories = {}  # Use a dictionary to track unique categories with increasing IDs
    annotations = []
    category_id = 1  # Track category ID

    # Find all the species in the image directory, regardless of the dataset split
    for filename in listdir(image_folder):
      if not filename.endswith((".jpg", ".png")):  # Check for image extensions
          continue
        
      # Extract information from filename
      parts = filename.split("_")
      observation_id, family, genus, species = parts[:4]
      
      if genus == "nan":
        continue
  
      # Create category entry (if not already present)
      category_key = f"{family}_{genus}_{species}"  # Key for unique category identification
      if category_key not in categories:
        categories[category_key] = category_id  # Assign unique ID
        category_id += 1  # Increment ID for next category


    # Process each image in the list
    for filename in image_list:
        if not filename.endswith((".jpg", ".png")):  # Check for image extensions
          continue
        
        # Extract information from filename
        filename_with_ext = os.path.basename(filename)
        base_filename, _ = splitext(filename_with_ext)
        parts = base_filename.split("_")
        observation_id, family, genus, species = parts[:4]
        category_key = f"{family}_{genus}_{species}"  # Key for unique category identification
        
        if genus == "nan":
          print(f"Warning: Skipping image {base_filename} - incorrect format")
          continue
        
        # Create image entry
        image_id = len(images) + 1
        image_data = {
            "id": image_id,
            "width": None,  # Replace with actual image width (obtain from library)
            "height": None,  # Replace with actual image height (obtain from library)
            "file_name": filename_with_ext,
        }
        images.append(image_data)
    
        # Create annotation entry using retrieved category ID
        annotation = {
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": categories[category_key],
        }
        annotations.append(annotation)
    
    # Final COCO data structure
    coco_data = {
        "info": info,
        "images": images,
        "categories": [{
            "id": cat_id,
            "species": taxon.split("_")[2],
            "genus": taxon.split("_")[1],
            "family": taxon.split("_")[0],
        } for taxon, cat_id in categories.items()],  # Convert category IDs to full entries
        "annotations": annotations,
    }
      
    # Write COCO data to JSON file
    with open(output_filename, 'w') as outfile:
      json.dump(coco_data, outfile, indent=4)
      
    print(f"COCO annotations saved to: {output_filename}")
  
# Example usage (replace 'path/to/images' and 'output.json' with your actual paths)
create_coco_annotations(image_sets['train'], folder_dir, "train_neats.json")
create_coco_annotations(image_sets['validation'], folder_dir, "val_neats.json")
create_coco_annotations(image_sets['test'], folder_dir, "test_neats.json")


def load_and_print_annotations(filename, num_samples=10):
  """
  Loads a COCO annotation JSON file and prints the first N annotations.

  Args:
    filename: Path to the COCO annotation JSON file.
    num_samples: Number of annotations to print (default: 10).
  """

  # Load JSON data
  with open(filename, 'r') as infile:
    coco_data = json.load(infile)

  # Print first N annotations
  print("Total number of samples: ", len(coco_data["annotations"]))
  # Print first N annotations
  print("Total number of categories: ", len(coco_data["categories"]))
  # Print first N annotations
  print("First", num_samples, "annotations:")
  for i in range(min(num_samples, len(coco_data["annotations"]))):
    annotation = coco_data["annotations"][i]

    # Find image and category information based on IDs
    image_id = annotation["image_id"]
    image_data = None
    for image in coco_data["images"]:
      if image["id"] == image_id:
        image_data = image
        break

    category_id = annotation["category_id"]
    category = None
    for cat in coco_data["categories"]:
      if cat["id"] == category_id:
        category = cat
        break

    # Print annotation information
    print(f"\t- Annotation ID: {annotation['id']}")
    if image_data:
      print(f"\t\tImage: {image_data['file_name']}")
    else:
      print("\t\tImage: Not found")
    if category:
      print(f"\t\tCategory:")
      print(f"\t\t\tID: {category['id']}")
      print(f"\t\t\tSpecies: {category['species']}")
      print(f"\t\t\tFamily: {category['family']}")
      print(f"\t\t\tGenus: {category['genus']}")
    else:
      print("\t\tCategory: Not found")

# Example usage (replace 'output.json' with your actual filename)
load_and_print_annotations("train_neats.json", 20)

