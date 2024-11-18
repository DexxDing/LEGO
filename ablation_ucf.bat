@echo off
echo Running model training scripts...

python main_go.py --dataset ucf --ablation go-matrix --batch-size 512

python main_go.py --dataset ucf --go fc --ablation go-fc --batch-size 512

python main_threshold.py --dataset ucf --ablation threshold --batch-size 512

python main_topk.py --dataset ucf  --ablation topk --batch-size 512

python main_w_var.py --dataset ucf --ablation wvar --batch-size 512

echo All jobs have been executed.
pause


