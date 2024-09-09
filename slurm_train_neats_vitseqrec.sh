#!/bin/bash
#SBATCH --account=def-phgig4
#SBATCH --nodelist=ul-val-pr-cpc02
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu_72h
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --job-name=job_vitb_hier
#SBATCH --output=%x-%j.out

# GET ENVIRONMENT AND MODULES
# source ~/projects/myvenvs/torch1.13.1-cu11.7/bin/activate
# module load python/3.9
module load cuda/11.7
module load qt
module load geos
module load llvm
module load gcc
module load opencv
module load scipy-stack

source ~/projects/myvenvs/torch-2.0.1-cu117/bin/activate

# Variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MAKEFLAGS="-j$(nproc)"
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

NUM_PROC=4
shift


# # VANILLA
# # big 224x224
# torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
#     --model vit_base_patch16_224 \
#     --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_b16.pth" \
#     --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
#     --input-size 3 224 224 \
#     --batch-size 512 \
#     --grad-accum-steps 1 \
#     --num-classes 283 \
#     --lr-base 3e-3 \
#     --lr-base-scale 'linear' \
#     --epochs 150 \
#     --warmup-epochs 7 \
#     --drop-path 0.2 \
#     --mixup 0.1 \
#     --clip-grad 1.0 \
#     --weight-decay 0 \
#     --recovery-interval 0 \
#     --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
#     --amp \
#     --eval-metric "top1" \
#     --checkpoint-hist 3 \
#     --reprob 0.5 \
#    # --color-jitter 0.
#    # --experiment neats_478k \
#    # --log-wandb


# # SEQ
# # big 224x224
# torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
#     --model vit_base_seq_patch16_224 \
#     --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_b16.pth" \
#     --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
#     --input-size 3 224 224 \
#     --batch-size 512 \
#     --grad-accum-steps 1 \
#     --num-classes 283 \
#     --lr-base 3e-3 \
#     --lr-base-scale 'linear' \
#     --epochs 150 \
#     --warmup-epochs 7 \
#     --drop-path 0.2 \
#     --mixup 0.1 \
#     --clip-grad 1.0 \
#     --weight-decay 0 \
#     --recovery-interval 0 \
#     --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
#     --amp \
#     --eval-metric "species_acc" \
#     --checkpoint-hist 3 \
#     --reprob 0.5 \
#    # --experiment neats_478k \
#    # --log-wandb

# # big 512x512
# torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
#     --model vit_base_seq_patch16_224 \
#     --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_b16_512.pt" \
#     --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
#     --input-size 3 512 512 \
#     --batch-size 128 \
#     --grad-accum-steps 4 \
#     --num-classes 283 \
#     --lr-base 3e-3 \
#     --lr-base-scale 'linear' \
#     --epochs 150 \
#     --warmup-epochs 7 \
#     --drop-path 0.2 \
#     --mixup 0.1 \
#     --clip-grad 1.0 \
#     --weight-decay 0 \
#     --recovery-interval 0 \
#     --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
#     --amp \
#     --eval-metric "species_acc" \
#     --checkpoint-hist 3 \
#     --reprob 0.5 \
#    # --experiment neats_478k \
#    # --log-wandb

# torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
#     --model vit_huge_seq_patch14_224 \
#     --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_h14.pt" \
#     --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
#     --input-size 3 224 224 \
#     --batch-size 64 \
#     --grad-accum-steps 8 \
#     --num-classes 287 \
#     --lr-base 3e-3 \
#     --lr-base-scale 'linear' \
#     --epochs 150 \
#     --warmup-epochs 7 \
#     --drop-path 0.2 \
#     --mixup 0.1 \
#     --clip-grad 1.0 \
#     --weight-decay 0 \
#     --recovery-interval 0 \
#     --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
#     --amp \
#     --eval-metric "species_acc" \
#     --checkpoint-hist 3 \
#     --reprob 0.5 \
#    # --experiment neats_478k \
#    # --log-wandb

# Huge 512x512
torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
    --model vit_huge_seq_patch14_224 \
    --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_h14_518.pt" \
    --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
    --input-size 3 518 518 \
    --batch-size 24 \
    --grad-accum-steps 21 \
    --num-classes 283 \
    --lr-base 3e-3 \
    --lr-base-scale 'linear' \
    --epochs 150 \
    --warmup-epochs 7 \
    --drop-path 0.2 \
    --mixup 0.1 \
    --clip-grad 1.0 \
    --weight-decay 0 \
    --recovery-interval 0 \
    --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
    --amp \
    --eval-metric "species_acc" \
    --checkpoint-hist 3 \
    --reprob 0.5 \
   # --experiment neats_478k \
   # --log-wandb

#    # --batch-size 7
#    # --grad-accum-steps 72

# VANILLA NAIVE HIER
# # big 224x224
# torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
#     --model vit_base_hier_naive_patch16_224 \
#     --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_b16.pth" \
#     --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
#     --input-size 3 224 224 \
#     --batch-size 512 \
#     --grad-accum-steps 1 \
#     --num-classes 999 \
#     --lr-base 3e-3 \
#     --lr-base-scale 'linear' \
#     --epochs 150 \
#     --warmup-epochs 7 \
#     --drop-path 0.2 \
#     --mixup 0.1 \
#     --clip-grad 1.0 \
#     --weight-decay 0 \
#     --recovery-interval 0 \
#     --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
#     --amp \
#     --eval-metric "species_acc" \
#     --checkpoint-hist 3 \
#     --reprob 0.5 \
#    # --color-jitter 0.
#    # --experiment neats_478k \
#    # --log-wandb

# # big 512x512
# torchrun --nproc_per_node=$NUM_PROC train_neats_vitseq_rec.py \
#     --model vit_base_hier_naive_patch16_224 \
#     --initial-checkpoint "/home/ulaval.ca/vigro7/projects/def-phgig4/models/maws_vit_b16_512.pt" \
#     --data-dir "/home/ulaval.ca/vigro7/projects/def-phgig4/neats/" \
#     --input-size 3 512 512 \
#     --batch-size 128 \
#     --grad-accum-steps 4 \
#     --num-classes 999 \
#     --lr-base 3e-3 \
#     --lr-base-scale 'linear' \
#     --epochs 150 \
#     --warmup-epochs 7 \
#     --drop-path 0.2 \
#     --mixup 0.1 \
#     --clip-grad 1.0 \
#     --weight-decay 0 \
#     --recovery-interval 0 \
#     --output "/home/ulaval.ca/vigro7/projects/def-phgig4/output/" \
#     --amp \
#     --eval-metric "species_acc" \
#     --checkpoint-hist 3 \
#     --reprob 0.5 \
#    # --color-jitter 0.
#    # --experiment neats_478k \
#    # --log-wandb
