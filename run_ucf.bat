@echo off
echo Running model training scripts...


python main_go.py --dataset ucf --feature-group both --modal ft --batch-size 1024

python main_go.py --dataset ucf --feature-group both  --modal av --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal at --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal fv+ft --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal fv*ft --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal fv(cat)ft --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal av+at --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal av*at --batch-size 1024

python main_go.py --dataset ucf --feature-group both --modal av(cat)at --batch-size 1024

echo All jobs have been executed.
pause