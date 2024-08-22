python train_pgd.py --batch-size 1024 --epochs 100 --attack-iters 10 --restarts 1 --out-dir pgd_10
python train_alg2.py --batch-size 1024 --epochs 80 --attack-iters 10 --restarts 1 --out-dir alg2_10

python train_pgd.py --batch-size 1024 --epochs 100 --attack-iters 20 --restarts 5 --out-dir pgd_20
python train_alg2.py --batch-size 1024 --epochs 150 --attack-iters 20 --restarts 5 --out-dir alg2_20

