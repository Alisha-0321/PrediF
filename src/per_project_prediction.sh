#Flakify's flaky test prediction using per-project validation
#dataset=$1
#data_path="../dataset"
data_path="../dataset/All-Pairs-Per-Project/predicting-flakies/Balanced/Balanced_no_Comments"
#data_path="../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_@Test/VC"

pair_name=$(echo $1 | cut -d'_' -f1)
results_path="../flaky_pair_per_project"
dataset_file="${data_path}/${1}.csv" 
model_weights="${results_path}/Per_project_model_weights_on_${1}_dataset/model_weights"
results_file="${results_path}/per_project_validation_results_on_${1}_dataset"

if [[ ! -d "${model_weights}" ]]; then
    mkdir -p "${model_weights}"
fi

#Flakify_cross_validation_results_on_IDoFT_dataset_Flakify.csv

if [[ $2 == "train" ]]; then
    if [[ ${pair_name} == "vpCombis" ]]; then
        python3 -W ignore cross_validation_per_project_predictor_bert.py $dataset_file $model_weights "${results_file}.csv" "Victim_Polluter_Pair_Per_project-Data"
    elif [[ ${pair_name} == "vcCombis" ]]; then
        python3 -W ignore cross_validation_per_project_predictor_bert.py $dataset_file $model_weights "${results_file}.csv" "Victim_Cleaner_Polluter_Pair_Per_project-Data"
    elif [[ ${pair_name} == "bssCombis" ]]; then
        python3 -W ignore cross_validation_per_project_predictor_bert.py $dataset_file $model_weights "${results_file}.csv" "Brittle_State-Setter_Pair_Per_project-Data"
    fi

elif [[ $2 == "test" ]]; then
    if [[ ${pair_name} == "vpCombis" ]]; then
        python3 -W ignore Testing_per_project_bert.py $dataset_file $model_weights "${results_file}.csv" "Victim_Polluter_Pair_Per_project-Data"
    elif [[ ${pair_name} == "vcCombis" ]]; then
        python3 -W ignore Testing_per_project_bert.py $dataset_file $model_weights "${results_file}.csv" "Victim_Cleaner_Polluter_Pair_Per_project-Data"
    elif [[ ${pair_name} == "bssCombis" ]]; then
        python3 -W ignore Testing_per_project_bert.py $dataset_file $model_weights "${results_file}.csv" "Brittle_State-Setter_Pair_Per_project-Data"
    fi

elif [[ $2 == "explain" ]]; then

    if [[ ${pair_name} == "vpCombis" ]]; then
        python3 -W ignore explainable_per_project.py $dataset_file $model_weights "${results_file}.csv" "Victim_Polluter_Pair_Per_project-Data"
    elif [[ ${pair_name} == "vcCombis" ]]; then
        python3 -W ignore explainable_per_project.py $dataset_file $model_weights "${results_file}.csv" "Victim_Cleaner_Polluter_Pair_Per_project-Data"
    elif [[ ${pair_name} == "bssCombis" ]]; then
        python3 -W ignore explainable_per_project.py $dataset_file $model_weights "${results_file}.csv" "Brittle_State-Setter_Pair_Per_project-Data"
    fi
fi


