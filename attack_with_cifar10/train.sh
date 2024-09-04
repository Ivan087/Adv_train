# python train_pgd.py --batch-size 1024 --epochs 100 --attack-iters 10 --restarts 1 --out-dir pgd_10
# python train_alg2.py --batch-size 1024 --epochs 80 --attack-iters 10 --restarts 1 --out-dir alg2_10

# python train_pgd.py --batch-size 1024 --epochs 100 --attack-iters 20 --restarts 5 --out-dir pgd_20
# python train_alg2.py --batch-size 1024 --epochs 150 --attack-iters 20 --restarts 5 --out-dir alg2_20


# python train_fgsm.py --batch-size 1024 --epochs 100 --dataset fashionmnist --out-dir fashion_fgsm
# python train_alg.py --batch-size 1024 --epochs 100 --dataset fashionmnist --out-dir fashion_alg
# python train_alg_reg.py --batch-size 512 --epochs 100 --dataset fashionmnist --out-dir fashion_alg_reg


python train_fgsm.py --batch-size 1024 --epochs 100 --dataset cifar10 --out-dir cifar_fgsm
python train_alg.py --batch-size 1024 --epochs 100 --dataset cifar10 --out-dir cifar_alg
python train_alg_reg.py --batch-size 512 --epochs 100 --dataset cifar10 --out-dir cifar_alg_reg

