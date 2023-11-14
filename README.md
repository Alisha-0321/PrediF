This repository contains two approaches to predict the tests that an order-dependent test may be dependent on. To run PrediF_L, follow the following steps,

# PrediF_L

```shell
cd PrediF_L/
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

Step-1: 
```shell
cd PrediF_O/
```

Step-2: 

Download the zip and extract it from the link in "10_OTO.txt" file. This contains all the 10 Orders of 1000 Test Orders(10 OTO's).

Step-3:

To setup the environment to run PrediF_O for 10 OTO's, run the following command.
```shell
bash setup.sh
```

Step-4:

For running PrediF_O for a specific OTO:
Let's say you are running PrediF_O for 2nd OTO, for #Methods heuristic, for Brittle-Statesetters, run the following commands.
```shell
cd Results 
cd 2 #for 2nd OTO
cd "Ranking - #Methods/" #for #Methods heuristic
cd "BSS/" #for Brittle-Statesetters
python3 getStat.py #to run PrediF_O
```


