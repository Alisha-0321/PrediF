For running 10 fold cross validation, run the following command.

```shell
bash run_train_and_inference.sh vpCombis_balanced "train"
```

```shell
bash run_train_and_inference.sh vpCombis_balanced "BERT"
```

For running per-project evaluation, run the following command.

```shell
bash per_project_prediction.sh vpCombis_balanced "train" 
```

```shell
bash per_project_prediction.sh vpCombis_balanced "test"
```
