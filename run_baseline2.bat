@echo off
python main.py --dataset ped2 --feature-group both --modal fvxft --batch-size 8
python main.py --dataset shanghai_v2 --feature-group both --modal fvxft --batch-size 32
python main.py --dataset ave --feature-group both --modal fvxft --batch-size 8
python main.py --dataset street --feature-group both --modal fvxft --batch-size 8
python main.py --dataset combine --feature-group both --modal fvxft --batch-size 8
echo Part 2 complete.
pause
