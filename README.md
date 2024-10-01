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

### To run <em>fv </em> :
`python main.py --dataset ped2 --feature-group both --modality fv --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality fv --batch-size 32`

`python main.py --dataset ave --feature-group both --modality fv --batch-size 8`

`python main.py --dataset street --feature-group both --modality fv --batch-size 8`

`python main.py --dataset combine --feature-group both --modality fv --batch-size 8`

### To run <em>ft </em> :
`python main.py --dataset ped2 --feature-group both --modality ft --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality ft --batch-size 32`

`python main.py --dataset ave --feature-group both --modality ft --batch-size 8`

`python main.py --dataset street --feature-group both --modality ft --batch-size 8`

`python main.py --dataset combine --feature-group both --modality ft --batch-size 8`


### To run <em>av </em>:
`python main.py --dataset ped2 --feature-group both  --modality av --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both  --modality av --batch-size 32`

`python main.py --dataset ave --feature-group both  --modality av --batch-size 8`

`python main.py --dataset street --feature-group both  --modality av --batch-size 8`

`python main.py --dataset combine --feature-group both  --modality av --batch-size 8`


### To run <em>at </em> :
`python main.py --dataset ped2 --feature-group both --modality at --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality at --batch-size 32`

`python main.py --dataset ave --feature-group both --modality at --batch-size 8`

`python main.py --dataset street --feature-group both --modality at --batch-size 8`

`python main.py --dataset combine --feature-group both --modality at --batch-size 8`


### To run <em>fv + ft</em> :
`python main.py --dataset ped2 --feature-group both --modality fv+ft --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality fv+ft --batch-size 32`

`python main.py --dataset ave --feature-group both --modality fv+ft --batch-size 8`

`python main.py --dataset street --feature-group both --modality fv+ft --batch-size 8`

`python main.py --dataset combine --feature-group both --modality fv+ft --batch-size 8`

### To run <em>fv * ft</em> :
`python main.py --dataset ped2 --feature-group both --modality fv*ft --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality fv*ft --batch-size 32`

`python main.py --dataset ave --feature-group both --modality fv*ft --batch-size 8`

`python main.py --dataset street --feature-group both --modality fv*ft --batch-size 8`

`python main.py --dataset combine --feature-group both --modality fv*ft --batch-size 8`

### To run <em>fv (cat) ft</em> :
`python main.py --dataset ped2 --feature-group both --modality fv(cat)ft --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality fv(cat)ft --batch-size 32`

`python main.py --dataset ave --feature-group both --modality fv(cat)ft --batch-size 8`

`python main.py --dataset street --feature-group both --modality fv(cat)ft --batch-size 8`

`python main.py --dataset combine --feature-group both --modality fv(cat)ft --batch-size 8`


### To run <em>av + at</em> :
`python main.py --dataset ped2 --feature-group both --modality av+at --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality av+at --batch-size 32`

`python main.py --dataset ave --feature-group both --modality av+at --batch-size 8`

`python main.py --dataset street --feature-group both --modality av+at --batch-size 8`

`python main.py --dataset combine --feature-group both --modality av+at --batch-size 8`


### To run <em>av*at</em> :

`python main.py --dataset ped2 --feature-group both --modality av*at --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality av*at --batch-size 32`

`python main.py --dataset ave --feature-group both --modality av*at --batch-size 8`

`python main.py --dataset street --feature-group both --modality av*at --batch-size 8`

`python main.py --dataset combine --feature-group both --modality av*at --batch-size 8`


### To run <em>av(cat)at</em> :
`python main.py --dataset ped2 --feature-group both --modality av(cat)at --batch-size 8`

`python main.py --dataset shanghai_v2 --feature-group both --modality av(cat)at --batch-size 32`

`python main.py --dataset ave --feature-group both --modality av(cat)at --batch-size 8`

`python main.py --dataset street --feature-group both --modality av(cat)at --batch-size 8`

`python main.py --dataset combine --feature-group both --modality av(cat)at --batch-size 8`


