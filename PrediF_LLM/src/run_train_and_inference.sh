#Flakify's flaky test prediction using cross-validation

data_path="../dataset/All-Pairs-Per-Project/predicting-flakies/Balanced/Balanced_no_Comments"
#data_path="../dataset/All-Pairs-Per-Project/predicting-flakies/Balanced/Balanced_Org"
#data_path="../dataset" # For method coverage
pair_name=$(echo $1 | cut -d'_' -f1)
results_path="../${pair_name}_pair"
dataset_file="${data_path}/$1.csv" 
model_weights="${results_path}/model_weights_on_${1}_dataset" #For saving the model.  Model takei weight bole. For each model, we will save one best model
results_file="${results_path}/results_on_${1}" 

if [[ ! -d "${results_path}" ]]; then
    mkdir -p "${results_path}"
fi
#Flakify_cross_validation_results_on_IDoFT_dataset_Flakify.csv
if [[ $2 == "train" ]]; then
    if [[ ${pair_name} == "vpCombis" ]]; then
        #python3 Bert_train.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Polluter_Pair-Data"
        #../dataset/All-Pairs-Per-Project/predicting-flakies/Balanced/Balanced_no_Comments/vpCombis_balanced.csv
        echo $dataset_file
        python3 Bert_graphcodebert.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Polluter_Pair-Data" # runnable
        #python3 Bert_train_codebert.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Polluter_Pair-Data"
        #python3 Bert_train_with_chunks.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Polluter_Pair-Data"
        #python3 CodeT5_train.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Polluter_Pair-Data-CodeT5"
    elif [[ ${pair_name} == "vcCombis" ]]; then
        #python3 Bert_train.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Cleaner_Polluter_Pair-Data"
        python3 Bert_graphcodebert.py $dataset_file $model_weights "${results_file}_BERT.csv" "Victim_Cleaner_Polluter_Pair-Data" # runnable
    elif [[ ${pair_name} == "bssCombis" ]]; then
        python3 Bert_graphcodebert.py $dataset_file $model_weights "${results_file}_BERT.csv" "Brittle_State-Setter_Pair-Data" # runnable
    fi

elif [[ $2 == "explain" ]]; then
    if [[ ${pair_name} == "vpCombis" ]]; then
        python3 explainable.py $dataset_file $model_weights "${results_file}_explain.csv" "Victim_Polluter_Pair-Data"
    elif [[ ${pair_name} == "vcCombis" ]]; then
        python3 explainable.py $dataset_file $model_weights "${results_file}_explain.csv" "Victim_Cleaner_Polluter_Pair-Data"
    elif [[ ${pair_name} == "bssCombis" ]]; then
        python3 explainable.py $dataset_file $model_weights "${results_file}_explain.csv" "Brittle_State-Setter_Pair-Data"
    fi
    #python3 explainable.py $dataset_file $model_weights "${results_file}_explain.csv" "IDOFT_Categorization-Data"
else
    if [[ $2 == "test-per-project" ]]; then
        echo "I AM GOING TO TEST Per-Project flaky-vs-nonflaky detection"
        results_file="${results_path}/sklearn_predictor/per_project_results_on_${1}_dataset" 
        python3 -W ignore cross_validation_per_project_predictor_Tml.py $dataset_file $model_weights "${results_file}.csv" $3
    else
        echo "ONLY for bert inference ****"
        if [[ ${pair_name} == "vpCombis" ]]; then
            echo $model_weights
            python3 Bert_inference.py $dataset_file $model_weights "${results_file}_$2_inference.csv" "Victim_Polluter_Pair-Data"
        elif [[ ${pair_name} == "vcCombis" ]]; then
            python3 Bert_inference.py $dataset_file $model_weights "${results_file}_$2_inference.csv" "Victim_Cleaner_Polluter_Pair-Data"
        elif [[ ${pair_name} == "bssCombis" ]]; then
            python3 Bert_inference.py $dataset_file $model_weights "${results_file}_$2_inference.csv" "Brittle_State-Setter_Pair-Data"
        fi
    fi
fi

