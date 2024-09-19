#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:39:56 2022

@author: vince
"""
import sys
import os

# Add local GitHub clone of timm to path
github_timm_path = os.path.expanduser('/home/vincent/repos/pytorch-image-models')
if github_timm_path not in sys.path:
    sys.path.insert(0, github_timm_path)
# Now import timm
import timm
# Check where timm is imported from
print(timm.__file__)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from contextlib import suppress
from functools import partial
import pycocotools.mask as mask_util
from tqdm import tqdm

from data import NEATS

from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets.coco import load_coco_json, convert_to_coco_json, convert_to_coco_dict
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.utils.file_io import PathManager

from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model
from timm.data.transforms_factory import create_transform

# from timm.models import ViTMG
# from timm.models import ViTSEQ
from models import ViTSEQREC
# from timm.models import ViTSEQRECPSM
# from timm.models import VisionTransformerHierNaive

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in dataset split, for IterableDatasets.')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
parser.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
parser.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')

parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrained-path', default=None, type=str,
                   help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--crop-border-pixels', type=int, default=None,
                    help='Crop pixels from image border.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=156,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=True,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')
parser.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
parser.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
parser.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')
parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')

_logger = setup_logger(name=__name__)

def xywh_to_x0y0x1y1(xywh_abs):
    x, y, w, h = xywh_abs
    x0 = x
    y0 = y
    x1 = x + w
    y1 = y + h
    return (x0, y0, x1, y1)

def decode_rle_to_tensor(rle):
    """
    Decode an RLE encoded mask to a PyTorch tensor.

    Args:
    - rle (dict): Run-Length Encoding of the mask.

    Returns:
    - mask_tensor (torch.Tensor): Decoded mask as a PyTorch tensor.
    """
    # Decode the RLE mask to a numpy array
    mask_array = mask_util.decode(rle)

    # Convert the numpy array to a PyTorch tensor
    mask_tensor = torch.from_numpy(mask_array).type(torch.bool)

    return mask_tensor

def crop_to_square_bottom(bbox):
    """
    Crops the bottom of a bounding box to make it a square, keeping the shortest edge.

    Parameters:
    bbox (tuple): A tuple (x0, y0, x1, y1) representing the bounding box coordinates.

    Returns:
    tuple: A tuple (x0, y0, x1, y1) representing the cropped square bounding box.
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    # Determine the shortest edge
    shortest_edge = min(width, height)

    # Adjust the bounding box to make it a square by reducing the larger dimension
    if width > height:
        new_x1 = x0 + shortest_edge
        new_y1 = y1  # keep the bottom edge as is
        return (x0, y1 - shortest_edge, new_x1, y1)
    else:
        new_y1 = y0 + shortest_edge
        new_x1 = x1  # keep the right edge as is
        return (x0, y0, x1, new_y1)


if __name__ == '__main__':
    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    # Create model
    if args.model=='vit_base_seq_rec_patch16':
        model = ViTSEQREC(
                        img_size=args.input_size[-2:],
                        patch_size=16,
                        in_chans=3,
                        num_classes=args.num_classes,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4.,
                        drop_rate=0.,
                        drop_path_rate=0.2,
                        pretrained_path=None,
                    )
        # Load checkpoint
        load_checkpoint(model, args.initial_checkpoint, args.use_ema)
    elif args.model=='vit_huge_seq_patch14':
        model = ViTSEQREC(
                        img_size=args.input_size[-2:],
                        patch_size=14,
                        in_chans=3,
                        num_classes=args.num_classes,
                        embed_dim=1280,
                        depth=32,
                        num_heads=16,
                        mlp_ratio=4.,
                        drop_rate=0.,
                        drop_path_rate=0.2,
                        pretrained_path=None,
                    )
        # Load checkpoint
        load_checkpoint(model, args.initial_checkpoint, args.use_ema)
    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            in_chans=in_chans,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            **factory_kwargs,
            **args.model_kwargs,
        )
        if args.head_init_scale is not None:
            with torch.no_grad():
                model.get_classifier().weight.mul_(args.head_init_scale)
                model.get_classifier().bias.mul_(args.head_init_scale)
        if args.head_init_bias is not None:
            nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

        if args.grad_checkpointing:
            model.set_grad_checkpointing(enable=True)

        # Load checkpoint
        load_checkpoint(model, args.initial_checkpoint, args.use_ema)


    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    # if args.num_classes is None:
    #     assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
    #     args.num_classes = model.num_classes
    #
    # load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)
    model.eval()  # Set model to evaluation mode

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    # Get classes from dataloader
    # Local
    data_root = "/dummy/path"
    ann_val_file = "/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/iNatTrees/test_neats.json"
    dataset_eval = NEATS(root=data_root, ann_file=ann_val_file, mode="validation", class_index_offset=0, full_info=True)
    topdown_dic = dataset_eval.topdown_mapper         # !!! Warning: mapper is only good for class_index_offset = 0
    category_mapper = dataset_eval.categories
    _logger.info(f"Taxonomic dic: {category_mapper}")
    _logger.info(f"Category mapper: {topdown_dic}")
    _logger.info(f"Number of tree species in class mapper: {len(category_mapper)}")

    # Class mapper for metadata catalog
    class_list = ['' for _ in range(283 +1)]   # empty list, +1 is for trees without taxonomic labels
    # Starting from the species, populate the entire class list
    for cat in category_mapper:
        class_list[cat["family_id"]] = cat["family"]
        class_list[cat["genus_id"]] = cat["genus"]
        class_list[cat["id"]] = cat["genus"] + " " + cat["species"]

    class_list[-1] = "tree"     # last class is trees without taxonomic labels

    # COCO
    # Drone
    # coco_test_filename =  '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/dataset_drone/annotations/dataset_drone_22.json'
    # img_dir = '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/dataset_drone/images'
    # coco_test_filename =  '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/dataset_placette/annotations/instances.json'
    # img_dir = '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/dataset_placette/images'
    coco_test_filename =  '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/drone_sherbrooke_20240901/annotations/instances.json'
    img_dir = '/mnt/e1e899db-64e7-4653-855e-c0cbffc9c675/data/datasets/drone_sherbrooke_20240901/images'

    class_list[-1] = "arbre"     # last class is trees without taxonomic labels
    # # # 102MP
    # coco_test_filename =  '/home/vincent/Documents/supervisely_labels/coco_with_taxo.json'
    # img_dir = '/home/vincent/Downloads/photos_arbres_vincent'

    # Load from COCO file and register dataset in D2
    test_set_name = "tree_set_102MP"
    dicts_test = load_coco_json(coco_test_filename, img_dir, test_set_name)     # extra_annotation_keys=['score']
    _logger.info("Done loading {} samples.".format(len(dicts_test)))
    tree_metadata = MetadataCatalog.get(test_set_name)
    del tree_metadata.thing_classes
    MetadataCatalog.get(test_set_name).set(thing_classes=class_list)
    tree_metadata = MetadataCatalog.get(test_set_name)

    # Define transformation for the classifier
    transform = transforms.Compose([
        transforms.Resize(1096, max_size=None, antialias=True), # 585
        transforms.CenterCrop(size=args.input_size[-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(transform)


    # Visualize annotations
    genus_precision = []
    species_precision = []
    genus_good_species_bad_precision = []
    # random.shuffle(dicts_test)  # shuffle samples
    with torch.inference_mode():
        for dic in tqdm(dicts_test[:]):
            image_file = os.path.splitext(dic["file_name"])[0] + os.path.splitext(dic["file_name"])[1].upper()
            img = utils.read_image(image_file, "RGB")

            # Visualization
            fig, axs = plt.subplots(1, 2, figsize=(6, 10), dpi=200)  # 1 row, 3 columns
            visualizer = Visualizer(img, metadata=tree_metadata, scale=0.25)
            gt_image = visualizer.draw_dataset_dict(dic)
            axs[0].imshow(gt_image.get_image())
            axs[0].axis('off')
            axs[0].set_title('Ground Truth')

            # Global noised mask (each segmentation mask is noised on the image
            # to avoid overlapping with other segmentation masks)
            tensor_img = torch.from_numpy(np.array(img))
            global_noised_mask = tensor_img.clone()
            for ann in dic["annotations"]:
                # Segment-out the background
                mask = decode_rle_to_tensor(ann['segmentation'])
                # Fill the mask with random pixel normally distributed
                global_noised_mask[mask] = torch.empty(global_noised_mask[mask].shape, dtype=torch.float32).normal_().to(torch.uint8)
                # Fill the mask with 0
                # global_noised_mask[mask] = torch.zeros(global_noised_mask[mask].shape, dtype=torch.float32).to(torch.uint8)

            # # Debug
            # plt.imshow(global_noised_mask)
            # plt.show()

            # Classifier predictions for each gt_instances
            for ann in dic["annotations"]:
                # Get gt_bbox to input to the classifier
                bbox_native = xywh_to_x0y0x1y1(ann['bbox'])
                # Convert the numpy image to PIL
                img_pil = Image.fromarray(img)

                # Mask-out if another segmentation mask is overlapping the current one
                # Instead of computing each IoUs, which is long for 102MP images,
                # Denoise the current mask in the global detection mask
                mask = decode_rle_to_tensor(ann['segmentation'])
                global_noised_mask_temp = global_noised_mask.clone()
                global_noised_mask_temp[mask] = tensor_img[mask]
                # Convert back to PIL image
                img_pil = Image.fromarray(global_noised_mask_temp.numpy())

                # Segment-out the background
                not_background = decode_rle_to_tensor(ann['segmentation'])
                # Ensure the mask is in the format [1, 1, H, W] for interpolation
                not_background = not_background.unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions and convert to float
                # Resize the mask
                not_background = F.interpolate(not_background, size=(img_pil.height, img_pil.width), mode='nearest')

                # Remove the added dimensions
                not_background = not_background.squeeze(0).squeeze(0).bool()  # Convert back to the original data type if necessary

                # Convert the PIL image to a PyTorch tensor
                img_tensor = torch.from_numpy(np.array(img_pil)).float()

                # Ensure the mask has the same number of channels as the image (3 for RGB)
                if len(not_background.shape) == 2:  # If mask is [H, W]
                    not_background = not_background.unsqueeze(2).expand(-1, -1, 3)  # Make it [H, W, 3]

                # Fill the background with random pixel normally distributed
                # img_tensor[~not_background] = torch.empty((img_tensor[~not_background].shape), dtype=torch.float32).normal_()

                # Convert the result back to a PIL image
                im_masked_background = Image.fromarray((img_tensor.cpu().numpy()).astype(np.uint8))

                # Crop the image using the provided bounding box coordinates
                squared_bbox = crop_to_square_bottom(bbox_native)
                img_cropped = im_masked_background.crop(bbox_native)     # img_pil
                # # Debug
                # img_cropped.show()

                x = transform(img_cropped).unsqueeze(0).half().to(device)

                if args.no_prefetcher:
                    target = target.to(device)
                    x = x.to(device)
                if args.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)

                # Compute output
                with amp_autocast():
                    # output, attn_weights = model(x, targets=None, topdown_dic=topdown_dic, isLatMask=False, istrain=False, isVisMask=True)
                    output = model(x)

                    # Visualize prediction
                    # Scale the model predictions to add up to 1
                    pred_scores = torch.softmax(output[:,2,:], dim=1).cpu().detach()
                    # pred_scores = torch.softmax(output, dim=1).cpu().detach()           # vanilla base

                top_3_probs, top_3_indices = torch.topk(pred_scores, 3)
                # Add text labels for top 3 classes and probabilities
                idx = np.clip(top_3_indices[0][0], 0, 174)
                # Get gt and pred values
                cat_id_gt = ann['category_id']
                cat_id_dt = idx.item()
                # Change the category in the dic, visualization purpose
                ann['category_id'] = cat_id_dt

                # Compute precision TODO make it a function
                cat_string_gt = class_list[cat_id_gt]
                cat_string_dt = class_list[cat_id_dt]
                # Split the string into genus and species
                taxo_gt = cat_string_gt.split(' ')
                taxo_dt = cat_string_dt.split(' ')
                # Last class is trees without taxonomic labels
                if cat_id_gt == len(class_list) - 1:
                    continue
                # Genus
                genus_gt = taxo_gt[0]
                genus_dt = taxo_dt[0]
                genus_precision.append(genus_gt == genus_dt)

                # Species
                if len(taxo_gt) == 2 and len(taxo_dt) == 2:
                    species_gt = taxo_gt[1]
                    species_dt = taxo_dt[1]
                    species_precision.append(species_gt == species_dt)

                    # Measure if it makes better mistakes, so in cases of
                    # wrong predicted species, the genus is still correct
                    if species_gt != species_dt:
                        genus_good_species_bad_precision.append(genus_gt == genus_dt)


            # Visualize detection + classification
            visualizer = Visualizer(img, metadata=tree_metadata, scale=0.25)
            gt_image = visualizer.draw_dataset_dict(dic)

            axs[1].imshow(gt_image.get_image())
            axs[1].axis('off')
            axs[1].set_title('Predictions')
            plt.show()

    # Print the precision
    # Check for division by zero
    if len(genus_precision) == 0:
        genus_precision.append(0)
    if len(species_precision) == 0:
        species_precision.append(0)
    _logger.info("Genus precision: {:.2f} on {} predictions.".format(100 * sum(genus_precision) / len(genus_precision),  len(genus_precision)))
    _logger.info("Species precision: {:.2f} on {} predictions.".format(100 * sum(species_precision) / len(species_precision), len(species_precision)))
    _logger.info("Genus precision when species is wrong: {:.2f} on {} predictions.".format(100 * sum(genus_good_species_bad_precision) / len(genus_good_species_bad_precision), len(genus_good_species_bad_precision)))




# Parameter for command lines
# Vanilla
# --model vit_base_patch16_224 --initial-checkpoint "/home/vincent/Downloads/checkpoint-778980-130.pth.tar" --batch-size 1 --num-classes 283 --input-size 3 224 224
# Vanilla hier naive
# --model vit_base_hier_naive_patch16_224 --initial-checkpoint "/home/vincent/Downloads/checkpoint-vitb_hier_naive-141.pth.tar" --batch-size 1 --num-classes 283 --input-size 3 224 224
# Vit SEQ REC
# --model vit_base_seq_rec_patch16 --initial-checkpoint "/home/vincent/Downloads/vitb224_pe-128.pth.tar" --batch-size 1 --num-classes 283 --input-size 3 224 224
# --model vit_base_seq_rec_patch16 --initial-checkpoint "/home/vincent/Downloads/checkpoint-139_b512_pe.pth.tar" --batch-size 1 --num-classes 283 --input-size 3 512 512
# vit seq
# --model vit_base_seq_patch16 --initial-checkpoint "/home/vincent/Downloads/checkpoint-768122-139.pth.tar" --batch-size 1 --num-classes 283 --input-size 3 224 224
# --model vit_huge_seq_patch14 --initial-checkpoint "/home/vincent/Downloads/checkpoint-11.pth.tar" --batch-size 1 --num-classes 283 --input-size 3 518 518
