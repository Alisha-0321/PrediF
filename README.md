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

These are the steps that are needed to be followed to run PrediF_O,

Step-1: ```shell
cd PrediF_O/
```

Step-2: Download the 


