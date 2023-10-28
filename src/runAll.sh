#!/bin/bash
#bash run_train_and_inference.sh vpCombis_balanced "train"
#bash run_train_and_inference.sh vpCombis_balanced "BERT"

#bash run_train_and_inference.sh vpCombis_balanced_method_coverage "train"
#bash run_train_and_inference.sh vpCombis_balanced_method_coverage "BERT"

#bash run_train_and_inference.sh vcCombis_balanced "train"
#bash run_train_and_inference.sh vcCombis_balanced "BERT"

#bash run_train_and_inference.sh vcCombis_balanced_method_coverage "train"
#bash run_train_and_inference.sh vcCombis_balanced_method_coverage "BERT"


#bash run_train_and_inference.sh bssCombis_balanced "train"
#bash run_train_and_inference.sh bssCombis_balanced "BERT"


#bash per_project_prediction.sh vpCombis_balanced "train" 
bash per_project_prediction.sh vpCombis_balanced "test"


#bash per_project_prediction.sh vcCombis_balanced "train"
#bash per_project_prediction.sh vcCombis_balanced "test"


#bash per_project_prediction.sh bssCombis_balanced "train"
#bash per_project_prediction.sh bssCombis_balanced "test"


#bash run_train_and_inference.sh vpCombis_balanced "explain" "Bert"
#bash run_train_and_inference.sh bssCombis_balanced "explain" "Bert"
#bash run_train_and_inference.sh vcCombis_balanced "explain" "Bert"

#bash per_project_prediction.sh vcCombis_balanced "explain" 
