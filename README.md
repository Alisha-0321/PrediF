This repository contains two approaches to predict the tests that an order-dependent test may be dependent on. To run PrediF_LLM, follow the following steps,

# PrediF_LLM

```shell
cd PrediF_LLM/
cd src/
```

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

# PrediF_O
