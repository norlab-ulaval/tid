import json
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_taxonomic_distribution(json_file):
    """
    Plots the distribution of division, family, genus, and species from a given JSON file.

    Args:
        json_file: Path to the JSON file.
    """
    # Load JSON data
    with open(json_file, 'r') as infile:
        coco_data = json.load(infile)

    # Initialize counters
    division_counts = defaultdict(int)
    family_counts = defaultdict(int)
    genus_counts = defaultdict(int)
    species_counts = defaultdict(int)

    # Count occurrences
    for annotation in coco_data["annotations"]:
        category = annotation["category_id"]
        division = coco_data["categories"][category]["division"]
        family = coco_data["categories"][category]["family"]
        genus = coco_data["categories"][category]["genus"]
        species = coco_data["categories"][category]["species"]

        division_counts[division] += 1
        family_counts[family] += 1
        genus_counts[genus] += 1
        species_counts[genus + " " + species] += 1

    # Sort counts
    sorted_division_counts = dict(sorted(division_counts.items(), key=lambda item: item[1]))
    sorted_family_counts = dict(sorted(family_counts.items(), key=lambda item: item[1]))
    sorted_genus_counts = dict(sorted(genus_counts.items(), key=lambda item: item[1]))
    sorted_species_counts = dict(sorted(species_counts.items(), key=lambda item: item[1]))

    # Plot distributions
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    # Division
    axs[0, 0].barh(list(sorted_division_counts.keys()), list(sorted_division_counts.values()), color='skyblue')
    axs[0, 0].set_title('Division Distribution', fontsize=10)
    axs[0, 0].set_xlabel('Count', fontsize=8)
    axs[0, 0].set_ylabel('Division', fontsize=8)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=6)

    # Family
    axs[0, 1].barh(list(sorted_family_counts.keys()), list(sorted_family_counts.values()), color='skyblue')
    axs[0, 1].set_title('Family Distribution', fontsize=10)
    axs[0, 1].set_xlabel('Count', fontsize=8)
    axs[0, 1].set_ylabel('Family', fontsize=8)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=6)

    # Genus
    axs[1, 0].barh(list(sorted_genus_counts.keys()), list(sorted_genus_counts.values()), color='skyblue')
    axs[1, 0].set_title('Genus Distribution', fontsize=10)
    axs[1, 0].set_xlabel('Count', fontsize=8)
    axs[1, 0].set_ylabel('Genus', fontsize=8)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=6)

    # Species
    axs[1, 1].bar(list(sorted_species_counts.keys()), list(sorted_species_counts.values()), color='skyblue')
    axs[1, 1].set_title('Species Distribution', fontsize=8)
    axs[1, 1].set_xlabel('Count', fontsize=8)
    axs[1, 1].set_ylabel('Species', fontsize=8)
    axs[1, 1].tick_params(axis='x', which='major', labelsize=6, rotation=90)

    plt.tight_layout()
    plt.show()

# Example usage
anno_dir = "/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/"
plot_taxonomic_distribution(anno_dir + "val_neats_2264k.json")
