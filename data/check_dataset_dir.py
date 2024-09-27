#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:02:58 2024

@author: vincent
"""
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def find_unidentified_images(image_folder):
    unidentified_images = []
    num_img_files = len(os.listdir(image_folder))
    print(num_img_files)

    for filename in tqdm(os.listdir(image_folder)):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that it is, in fact, an image
        except UnidentifiedImageError:
            unidentified_images.append(file_path)
            print(file_path)
        except Exception as e:
            print(f"An unexpected error occurred with file {file_path}: {e}")

    return unidentified_images

if __name__ == "__main__":
    folder_path = "/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats"
    unidentified_images = find_unidentified_images(folder_path)

    if unidentified_images:
        print("The following images could not be identified:")
        for image in unidentified_images:
            print(image)
    else:
        print("All images were successfully identified.")


# # List of filenames to be deleted
# filenames = [
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161140190_Rosaceae_Prunus_serotina_2_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188513030_Sapindaceae_Acer_rubrum_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649947_Fagaceae_Quercus_muehlenbergii_2_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179053070_Sapindaceae_Acer_saccharinum_2_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163448_Salicaceae_Populus_grandidentata_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163460_Juglandaceae_Carya_cordiformis_1_of_5.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161140190_Rosaceae_Prunus_serotina_1_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052656_Fabaceae_Robinia_pseudoacacia_2_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187618505_Sapindaceae_Aesculus_hippocastanum_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052694_Salicaceae_Populus_deltoides_4_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188513366_Malvaceae_Tilia_cordata_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87444338_Cupressaceae_Taxodium_distichum_2_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187618262_Sapindaceae_Acer_pseudoplatanus_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163394_Sapindaceae_Acer_negundo_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187618596_Fabaceae_Gleditsia_triacanthos_3_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187618280_Simaroubaceae_Ailanthus_altissima_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649926_Fagaceae_Quercus_alba_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/157554876_Fabaceae_Gleditsia_triacanthos_1_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187618596_Fabaceae_Gleditsia_triacanthos_2_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87324243_Sapindaceae_Acer_saccharum_3_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/174168975_Simaroubaceae_Ailanthus_altissima_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/157554846_Fabaceae_Cercis_canadensis_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87324243_Sapindaceae_Acer_saccharum_2_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052694_Salicaceae_Populus_deltoides_2_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87563569_Fagaceae_Quercus_virginiana_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188512804_Ulmaceae_Ulmus_minor_2_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161140190_Rosaceae_Prunus_serotina_3_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052694_Salicaceae_Populus_deltoides_3_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163239_Betulaceae_Betula_populifolia_1_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188512761_Fagaceae_Fagus_grandifolia_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649947_Fagaceae_Quercus_muehlenbergii_1_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052656_Fabaceae_Robinia_pseudoacacia_1_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188512804_Ulmaceae_Ulmus_minor_1_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/45825823_Pinaceae_Pinus_strobus_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87324243_Sapindaceae_Acer_saccharum_4_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188512656_Altingiaceae_Liquidambar_styraciflua_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/157554876_Fabaceae_Gleditsia_triacanthos_2_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188512804_Ulmaceae_Ulmus_minor_3_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87450968_Fagaceae_Quercus_virginiana_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188513151_Fagaceae_Quercus_rubra_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163229_Magnoliaceae_Liriodendron_tulipifera_2_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649586_Pinaceae_Pinus_strobus_3_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/157225339_Cannabaceae_Celtis_occidentalis_3_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649586_Pinaceae_Pinus_strobus_2_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87324243_Sapindaceae_Acer_saccharum_1_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/45816154_Pinaceae_Pinus_sylvestris_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161140138_Fagaceae_Quercus_velutina_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649791_Fagaceae_Fagus_grandifolia_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187618596_Fabaceae_Gleditsia_triacanthos_1_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052656_Fabaceae_Robinia_pseudoacacia_3_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/157225339_Cannabaceae_Celtis_occidentalis_1_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/85020542_Fagaceae_Quercus_falcata_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163229_Magnoliaceae_Liriodendron_tulipifera_1_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179053114_Nyssaceae_Nyssa_sylvatica_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/188513077_Pinaceae_Pinus_sylvestris_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/174169180_Rosaceae_Prunus_pensylvanica_1_of_1.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161163239_Betulaceae_Betula_populifolia_2_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/161140099_Salicaceae_Populus_tremuloides_4_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179053070_Sapindaceae_Acer_saccharinum_1_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/157225339_Cannabaceae_Celtis_occidentalis_2_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/87444338_Cupressaceae_Taxodium_distichum_1_of_2.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649947_Fagaceae_Quercus_muehlenbergii_3_of_3.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/179052694_Salicaceae_Populus_deltoides_1_of_4.jpg',
#     '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/neats/187649586_Pinaceae_Pinus_strobus_1_of_4.jpg',
# ]

# # Iterate through the list of filenames
# for filename in filenames:
#     # Check if the file exists
#     if os.path.exists(filename):
#         # Delete the file
#         # os.remove(filename)
#         print(f"Deleted: {filename}")
#     else:
#         print(f"File not found: {filename}")
