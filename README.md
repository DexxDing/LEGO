### Features are avaliable at:
https://anu365-my.sharepoint.com/:f:/g/personal/u7321641_anu_edu_au/Ei3gIDwIcTJGuQzDJq2Ztb8BH40jYilfxjOG_jMqz3Kr8Q?e=amKPiZ

### Dataset file structure
```
|-- save  
|   |-- Shanghai
|   |   |-- SH_ten_crop_i3d_v2
|   |   `-- sent_emb_n
|   |   `-- c3d
|   |-- UCSDped2
|   |   |-- ped2_ten_crop_i3d
|   |   `-- sent_emb_n
|   |-- Avenue
|   |   |-- ave_ten_crop_i3d_v2
|   |   `-- sent_emb_n
|   |-- Street
|   |   |-- street_ten_crop_i3d
|   |   `-- sent_emb_n
```

### Ablation for <em>variance regularization</em> coefficient:
`python main_w_var.py --dataset ped2 --feature-group both --ablation wvar --batch-size 8`

`python main_w_var.py --dataset shanghai_v2 --feature-group both --ablation wvar --batch-size 32`

`python main_w_var.py --dataset ave --feature-group both --ablation wvar --batch-size 8`

`python main_w_var.py --dataset street --feature-group both --ablation wvar --batch-size 8`

`python main_w_var.py --dataset combine --feature-group both --ablation wvar --batch-size 8`

### Ablation for <em>mean regularization</em> coefficient:
`python main_w_mean.py --dataset ped2 --feature-group both --ablation wmean --batch-size 8`

`python main_w_mean.py --dataset shanghai_v2 --feature-group both --ablation wmean --batch-size 32`

`python main_w_mean.py --dataset ave --feature-group both --ablation wmean --batch-size 8`

`python main_w_mean.py --dataset street --feature-group both --ablation wmean --batch-size 8`

`python main_w_mean.py --dataset combine --feature-group both --ablation wmean --batch-size 8`


### Ablation for <em>topk</em> value:
`python main_topk.py --dataset ped2 --feature-group both  --ablation topk --batch-size 8`

`python main_topk.py --dataset shanghai_v2 --feature-group both  --ablation topk --batch-size 32`

`python main_topk.py --dataset ave --feature-group both  --ablation topk --batch-size 8`

`python main_topk.py --dataset street --feature-group both  --ablation topk --batch-size 8`

`python main_topk.py --dataset combine --feature-group both  --ablation topk --batch-size 8`


### Ablation for <em>threshold</em> value:
`python main_threshold.py --dataset ped2 --feature-group both --ablation threshold --batch-size 8`

`python main_threshold.py --dataset shanghai_v2 --feature-group both --ablation threshold --batch-size 32`

`python main_threshold.py --dataset ave --feature-group both --ablation threshold --batch-size 8`

`python main_threshold.py --dataset street --feature-group both --ablation threshold --batch-size 8`

`python main_threshold.py --dataset combine --feature-group both --ablation threshold --batch-size 8`


### Ablation for <em>graph operator(matrix form)</em> dimension:
`python main_go.py --dataset ped2 --feature-group both --ablation go-matrix --batch-size 8`

`python main_go.py --dataset shanghai_v2 --feature-group both --ablation go-matrix --batch-size 32`

`python main_go.py --dataset ave --feature-group both --ablation go-matrix --batch-size 8`

`python main_go.py --dataset street --feature-group both --ablation go-matrix --batch-size 8`

`python main_go.py --dataset combine --feature-group both --ablation go-matrix --batch-size 8`

### Ablation for <em>graph operator(fc form)</em> dimension:
`python main_go.py --dataset ped2 --feature-group both --go fc --ablation go-fc --batch-size 8`

`python main_go.py --dataset shanghai_v2 --feature-group both --go fc --ablation go-fc --batch-size 32`

`python main_go.py --dataset ave --feature-group both --go fc --ablation go-fc --batch-size 8`

`python main_go.py --dataset street --feature-group both --go fc --ablation go-fc --batch-size 8`

`python main_go.py --dataset combine --feature-group both --go fc --ablation go-fc --batch-size 8`
