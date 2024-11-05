# python train_pgd.py --batch-size 1024 --epochs 100 --attack-iters 10 --restarts 1 --out-dir pgd_10
# python train_alg2.py --batch-size 1024 --epochs 80 --attack-iters 10 --restarts 1 --out-dir alg2_10

# python train_pgd.py --batch-size 1024 --epochs 100 --attack-iters 20 --restarts 5 --out-dir pgd_20
# python train_alg2.py --batch-size 1024 --epochs 150 --attack-iters 20 --restarts 5 --out-dir alg2_20


# python train_fgsm.py --batch-size 1024 --epochs 100 --dataset fashionmnist --out-dir fashion_fgsm
# python train_alg.py --batch-size 1024 --epochs 100 --dataset fashionmnist --out-dir fashion_alg
# python train_alg_reg.py --batch-size 512 --epochs 100 --dataset fashionmnist --out-dir fashion_alg_reg


# python train_fgsm.py --batch-size 1024 --epochs 100 --dataset cifar10 --out-dir cifar_fgsm
# python train_alg.py --batch-size 1024 --epochs 100 --dataset cifar10 --out-dir cifar_alg
# python train_alg_reg.py --batch-size 512 --epochs 100 --dataset cifar10 --out-dir cifar_alg_reg
 
###################### FGSM and SSA ####################
# python train_ssa.py --batch-size 1024 --epochs 20 --dataset fashionmnist --out-dir out_fgsm_fashion
# python train_ssa.py --batch-size 1024 --epochs 20 --dataset cifar10 --out-dir out_fgsm_cifar
# python train_ssa.py --batch-size 512 --epochs 20 --dataset tinyimagenet --out-dir out_fgsm_tiny

# python train_ssa.py --batch-size 1024 --epochs 20 --dataset fashionmnist --ssa --out-dir out_fgsm_ssa_fashion
# python train_ssa.py --batch-size 1024 --epochs 20 --dataset cifar10 --ssa --out-dir out_fgsm_ssa_cifar
# python train_ssa.py --batch-size 512 --epochs 20 --dataset tinyimagenet --ssa --out-dir out_fgsm_ssa_tiny

##################### Free and SSA #####################
python train_free_ssa.py --out-dir out_free_fashion --dataset fashionmnist --batch-size 1024 --epochs 20 
python train_free_ssa.py --out-dir out_free_cifar --dataset cifar10 --batch-size 1024 --epochs 20 
python train_free_ssa.py --out-dir out_free_tiny --dataset tinyimagenet --batch-size 512 --epochs 20 

python train_free_ssa.py --out-dir out_free_ssa_fashion --dataset fashionmnist --batch-size 1024 --epochs 20 --ssa 
python train_free_ssa.py --out-dir out_free_ssa_cifar --dataset cifar10 --batch-size 1024 --epochs 20 --ssa
python train_free_ssa.py --out-dir out_free_ssa_tiny --dataset tinyimagenet --batch-size 512 --epochs 20 --ssa



