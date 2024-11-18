@echo off

python main_var.py --dataset shanghai_v2 --feature-group both --fusion add --aggregate_text --extra_loss --batch-size 32 

python main_mixloss.py --dataset shanghai_v2 --feature-group both --fusion add --aggregate_text --extra_loss --batch-size 32 

python main.py --dataset shanghai_v2 --feature-group both --fusion add --aggregate_text --extra_loss --batch-size 32 


echo All scripts have been executed.
pause
