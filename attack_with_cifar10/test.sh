# python test.py --model-path ./mnist_fgsm/model.pth --batch-size 1024 --dataset mnist --attack-iters 1 --attack-restarts 1
# python test.py --model-path ./mnist_alg/model.pth --batch-size 1024 --dataset mnist --attack-iters 1 --attack-restarts 1 
# python test.py --model-path ./mnist_alg_reg/model.pth --batch-size 1024 --dataset mnist --attack-iters 1 --attack-restarts 1

# python test.py --model-path ./mnist_fgsm/model.pth --batch-size 1024 --dataset mnist --attack-iters 10 --attack-restarts 1
# python test.py --model-path ./mnist_alg/model.pth --batch-size 1024 --dataset mnist --attack-iters 10 --attack-restarts 1
# python test.py --model-path ./mnist_alg_reg/model.pth --batch-size 1024 --dataset mnist --attack-iters 10 --attack-restarts 1

# python test.py --model-path ./mnist_fgsm/model.pth --batch-size 1024 --dataset mnist --attack-iters 20 --attack-restarts 5 
# python test.py --model-path ./mnist_alg/model.pth --batch-size 1024 --dataset mnist --attack-iters 20 --attack-restarts 5 
# python test.py --model-path ./mnist_alg_reg/model.pth --batch-size 1024 --dataset mnist --attack-iters 20 --attack-restarts 5 


# python test.py --model-path ./fashion_fgsm/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 1 --attack-restarts 1
# python test.py --model-path ./fashion_alg/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 1 --attack-restarts 1 
# python test.py --model-path ./fashion_alg_reg/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 1 --attack-restarts 1

# python test.py --model-path ./fashion_fgsm/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 10 --attack-restarts 1
# python test.py --model-path ./fashion_alg/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 10 --attack-restarts 1
# python test.py --model-path ./fashion_alg_reg/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 10 --attack-restarts 1

# python test.py --model-path ./fashion_fgsm/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack-restarts 5 
# python test.py --model-path ./fashion_alg/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack-restarts 5 
# python test.py --model-path ./fashion_alg_reg/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack-restarts 5 

# python test.py --model-path ./cifar_fgsm/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 1 --attack-restarts 1
# python test.py --model-path ./cifar_alg/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 1 --attack-restarts 1 
# python test.py --model-path ./cifar_alg_reg/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 1 --attack-restarts 1

# python test.py --model-path ./cifar_fgsm/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 10 --attack-restarts 1
# python test.py --model-path ./cifar_alg/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 10 --attack-restarts 1
# python test.py --model-path ./cifar_alg_reg/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 10 --attack-restarts 1

# python test.py --model-path ./cifar_fgsm/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack-restarts 5 
# python test.py --model-path ./cifar_alg/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack-restarts 5 
# python test.py --model-path ./cifar_alg_reg/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack-restarts 5 

# python test.py --model-path ./tiny_ssa/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 10 --attack bim --out-logfile tmp.log


############ FGSM and SSA ############
python test.py --model-path ./out_fgsm_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_fgsm_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_fgsm_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack fgsm --out-logfile tmp.log

python test.py --model-path ./out_fgsm_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_fgsm_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_fgsm_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack pgd --out-logfile tmp.log

python test.py --model-path ./out_fgsm_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_fgsm_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_fgsm_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_fgsm_ssa_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack bim --out-logfile tmp.log

############ Free and SSA #######
python test.py --model-path ./out_free_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_free_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_free_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack fgsm --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack fgsm --out-logfile tmp.log

python test.py --model-path ./out_free_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_free_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_free_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack pgd --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack pgd --out-logfile tmp.log

python test.py --model-path ./out_free_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_fashion/model.pth --batch-size 1024 --dataset fashionmnist --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_free_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_cifar/model.pth --batch-size 1024 --dataset cifar10 --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_free_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack bim --out-logfile tmp.log
python test.py --model-path ./out_free_ssa_tiny/model.pth --batch-size 512 --dataset tinyimagenet --attack-iters 20 --attack bim --out-logfile tmp.log


