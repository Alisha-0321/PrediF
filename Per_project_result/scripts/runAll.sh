# bash runAll.sh VP_project_list.csv isVictimPolluterPair
if [[ $1 == "" || $2 == "" ]]; then
    echo "project list (e.g., data/VP_all_module.csv)"
    echo "Pair type (e.g., isVictimPolluterPair)"
    echo "Give Top-K-Rank (e.g., 1/3/5)"
    exit
fi

if [[ ! -d "Result" ]]; then
    mkdir "Result"
fi

dir="../../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_Comments"

if [[ $2 == "isVictimPolluterPair" ]]; then
    resultForColumn1="Result/VP_On_10_Failing_Order_Result_Rank_$3.csv" # Victim count based on 3.5
    resultForColumn2="Result/VP_On_10_Failing_Order_Result.csv"  # Polluter_Rank
    result_runtime="Result/VP_Unbalanced_Runtime_$3.csv"
    prediction_runtime_filename="../Per-Victim-Runtime.txt" 

    if [[ -f $resultForColumn1 ]]; then
        rm $resultForColumn1
        rm $resultForColumn2
        rm $result_runtime
    fi
    echo "project,sha,module,victim,Runtime" >> $result_runtime
    echo "project,sha,module,victim,First_true_polluter_found_at,Positive_prediction(Class1)/Negative_prediction(Class0),fail_order,Pair-Name" >> "$resultForColumn2"

elif [[ $2 == "isBSSPair" ]]; then
    resultForColumn1="Result/BSS_On_10_Failing_Order_Result_Rank_$3.csv" # Victim count based on 3.5
    resultForColumn2="Result/BSS_On_10_Failing_Order_Result.csv"  # Polluter_Rank
    result_runtime="Result/BSS_Unbalanced_Runtime_$3.csv"
    prediction_runtime_filename="../Per-Brittle-Runtime.txt" 

    if [[ -f $resultForColumn1 ]]; then
        rm $resultForColumn1
        rm $resultForColumn2
        rm $result_runtime
    fi
    echo "project,sha,module,Brittle,Runtime" >> $result_runtime
    echo "project,sha,module,brittle,First_true_state-setter_found_at,Positive_prediction(Class1)/Negative_prediction(Class0),fail_order,Pair-Name" >> "$resultForColumn2"

elif [[ $2 == "isVictimPolluterCleanerPair" ]]; then
    #result="Result/VC_On_Unbalanced_Result_$3.csv"
    resultForColumn1="Result/VC_On_10_Failing_Order_Result_Rank_$3.csv" # Victim count based on 3.5
    resultForColumn2="Result/VC_On_10_Failing_Order_Result.csv"  # Polluter_Rank
    result_runtime="Result/VC_Unbalanced_Runtime_$3.csv"
    prediction_runtime_filename="../Per-Victim-Cleaner-Runtime.txt" 
    if [[ -f $resultForColumn1 ]]; then
        rm $resultForColumn1
        rm $resultForColumn2
        rm $result_runtime
    fi
    echo "project,sha,module,victim,Polluter,Runtime" >> $result_runtime
    echo "project,sha,module,victim,polluter,First_true_cleaner_found_at,Positive_prediction(Class1)/Negative_prediction(Class0),fail_order,Pair-Name" >> "$resultForColumn2"
fi



Total_Rank_Class0=0
Total_Rank_Class1=0
pair_count_Class0=0
pair_count_Class1=0
while read row
do  
    project=$(echo $row | cut -d',' -f1) 
    sha=$(echo $row | cut -d',' -f2) 
    module=$(echo $row | cut -d',' -f3) 
    
    proj=$(echo $project | sed 's/\//-/g')    #replace slash(/) into dash(-)
    proj_name_with_underscore=$(echo $project | sed 's/\//_/g')    #replace slash(/) into dash(-)
    modified_module=$(echo $module | sed 's/\//_/g')

    if [[ $modified_module == "" ]]; then
        modified_module="."
    fi

    if [[ $2 == "isVictimPolluterPair" ]]; then
        python3 collect_vp_vpc_bs_label.py "$dir/VP_Per_Victim/VP_${proj_name_with_underscore}.csv" #outputs X.txt
        type="Victim"
        start=1
        end=10
        fifty_percent=5
        total_orders=10
    elif [[ $2 == "isBSSPair" ]]; then
        python3 collect_vp_vpc_bs_label.py "$dir/BSS_Per_Brittle/BSS_${proj_name_with_underscore}.csv" #outputs X.txt
        type="Brittle"
        start=1
        end=10
        total_orders=10
        fifty_percent=5
    elif [[ $2 == "isVictimPolluterCleanerPair" ]]; then
        python3 collect_vp_vpc_bs_label.py "$dir/VC/VC_${proj_name_with_underscore}.csv" #outputs X.txt
        type="VictimPolluterCleaner"
        start=0
        end=0
        total_orders=1
        fifty_percent=1 #it is 100%
    fi

    grep ',1$' "X.txt" > "Result/all_one_with_all_module.csv"
    grep ",$module," "Result/all_one_with_all_module.csv" > "Result/All_positive_pairs_From_Original_Data.csv" #This is used only for getting victim or brtille names

    if [[  $2 == "isVictimPolluterPair"  || $2 == "isBSSPair" ]]; then 
        cut -d',' -f4 "Result/All_positive_pairs_From_Original_Data.csv" | sort | uniq > "Org_Only_$type.csv" 
    else
        cut -d',' -f4-5 "Result/All_positive_pairs_From_Original_Data.csv" | sort | uniq > "Org_Only_$type.csv" 
    fi
    
    #echo $module
    if [[ $module == "." ]]; then
        #echo "I get module ."
        mvn_overhead=$(grep -r "${proj_name_with_underscore}," "Result/Average_Overhead.csv" | rev | cut -d',' -f1 | rev) 
    else
        mvn_overhead=$(grep -r ",${module}," "Result/Average_Overhead.csv" | rev | cut -d',' -f1 | rev) 
    fi
    
    vic_count_for_a_module=0
    polluter_rank=0
    runtime_for_all_victim=0
    prediction_runtime_per_module=0
    total_prediction_runtime=0 
    total_positive_negative_pairs_per_module=0
    while read vic_or_brittle #uniq victim or brittle or victim-polluter
    do
        if [[ $vic_or_brittle == *","* ]]; then #This will occur for VPC
            second_test=$(echo "$vic_or_brittle" | cut -d',' -f2)
        else
            second_test=""
        fi
        first_test=$(echo $vic_or_brittle | cut -d',' -f1)
        if [[ $second_test == "" ]]; then
            test_class_and_meth=$(echo "${first_test}" | rev | cut -d'.' -f1-2 | rev)
            searching_key_for_prediction_runtime="${first_test},"
        else
            test_class_and_meth="$(echo "${first_test}" | rev | cut -d'.' -f1-2 | rev)<AND>$(echo "${second_test}" | rev | cut -d'.' -f1-2 | rev)"
            searching_key_for_prediction_runtime="${first_test}<AND>${second_test},"
        fi
        found_in_topN=0
        runtime_for_all_pairs_for_a_victim=0
        for fail_order in $(seq $start $end); do
            flag_from_class1=0
            file_name_class1="../Per-Proj-Result/${proj}/Ranking_$2_${modified_module}_${test_class_and_meth}_Class1_${fail_order}.txt"
            total_possitive_pair=$(wc -l < "$file_name_class1")
            sort -t',' -k1,1nr ${file_name_class1} > "Result/tmp_sorted_list_class1.txt" #Model's predicted positive-pair result 
            grep -r "$vic_or_brittle," "Result/tmp_sorted_list_class1.txt" > "tmp1" #This one will contain all the victim-polluter pair for a victim
            file_name_class0="../Per-Proj-Result/${proj}/Ranking_$2_${modified_module}_${test_class_and_meth}_Class0_${fail_order}.txt"
            sort -t',' -k1,1n ${file_name_class0} > "Result/tmp_sorted_list_class0.txt" #Model's predicted positive-pair result 
            grep -r "${vic_or_brittle}" "Result/tmp_sorted_list_class0.txt" > "tmp0"
            total_negative_pair=$(wc -l < "$file_name_class0")
            total_positive_negative_pairs_per_module=$((total_positive_negative_pairs_per_module + total_possitive_pair + total_negative_pair))
            rank=0
            while read tmp_line
            do
                rank=$((rank+1))
                pair=$(echo $tmp_line | cut -d"'" -f2)

                #FOR Runtime
                OLDIFS=$IFS
                IFS=','
                read -r -a pairItems <<< "$pair"  # This syntax avoids word splitting issues.
                IFS=$OLDIFS  # Restore IFS after you're done.
                for item in "${pairItems[@]}"; do
                
                    #runtime=$(grep ",$item," "Result/Each-Test-Actual-Runtime.csv" | rev | cut -d',' -f1 | rev) 
                    runtime=$(grep ",$item," "Result/merged_prevandbala_final_runtimes.csv" | rev | cut -d',' -f1 | rev) 
                    #if [[ $runtime == "" ]]; then
                    #    echo "I am missing="$item, $runtime
                    #else
                    #    runtime_per_victim=$(echo "scale=4; ($runtime_per_victim) + ($runtime) " | bc)
                    #fi
                    runtime_for_all_pairs_for_a_victim=$(echo "scale=2; ($runtime_for_all_pairs_for_a_victim) + ($runtime) + ($mvn_overhead)" | bc)
                done

                exists=$(grep -r ",${pair}," "Result/All_positive_pairs_From_Original_Data.csv" | wc -l) 
                if [[ $exists -gt 0 ]]; then
                    if [[ $rank -le $3 ]]; then
                        found_in_topN=$((found_in_topN + 1))
                    fi
                    echo "$row,$vic_or_brittle,$rank,Class1,$fail_order,$pair" >> "$resultForColumn2"
                    pair_count_Class1=$((pair_count_Class1 + 1))
                    Total_Rank_Class1=$((rank+Total_Rank_Class1))
                    flag_from_class1=1 
                    break
                fi
            done < "tmp1"

            if [[ $flag_from_class1  -eq 0 ]]; then # If the correct polluter not found from class1
                rank=0
                total_line_in_class1=$(wc -l < tmp1) 
                while read tmp_line
                do
                    rank=$((rank+1))
                    pair=$(echo $tmp_line | cut -d"'" -f2)

                    #FOR Runtime
                    OLDIFS=$IFS
                    IFS=','
                    read -r -a pairItems <<< "$pair"  # This syntax avoids word splitting issues.
                    IFS=$OLDIFS  # Restore IFS after you're done.
                    for item in "${pairItems[@]}"; do 
                        #runtime=$(grep ",$item," "Result/Each-Test-Actual-Runtime.csv" | rev | cut -d',' -f1 | rev) 
                        runtime=$(grep ",$item," "Result/merged_prevandbala_final_runtimes.csv" | rev | cut -d',' -f1 | rev) 
                        #if [[ $runtime == "" ]]; then
                        #    echo "I am missing="$item, $runtime
                        #else
                        #    runtime_per_victim=$(echo "scale=4; ($runtime_per_victim) + ($runtime) " | bc)
                        #fi
                        runtime_for_all_pairs_for_a_victim=$(echo "scale=2; ($runtime_for_all_pairs_for_a_victim) + ($runtime) + ($mvn_overhead) " | bc)
                    done

                    exists=$(grep -r ",${pair}," "Result/All_positive_pairs_From_Original_Data.csv" | wc -l)
                    if [[ $exists -gt 0 ]]; then
                        rank=$((rank + total_line_in_class1))
                        if [[ $rank -le $3 ]]; then
                            found_in_topN=$((found_in_topN + 1))
                        fi
                        echo "$row,$vic_or_brittle,$rank,Class0,$fail_order,$pair" >> "$resultForColumn2"
                        pair_count_Class0=$((pair_count_Class0 + 1))
                        Total_Rank_Class0=$((rank+Total_Rank_Class0))
                        break
                    fi

                done < "tmp0"

            fi


            #Compute the rank of polluter
            polluter_rank=$((polluter_rank + rank)) #Total_polluter_rank accross all 10 orders and all victims

            if [[ -f "tmp1" ]]; then
                rm "Result/tmp_sorted_list_class1.txt"
                rm "tmp1"
            fi

            if [[ -f "tmp0" ]]; then
                rm "Result/tmp_sorted_list_class0.txt"
                rm "tmp0"
            fi

        done # End of 10 orders
        avg_runtime_per_order_for_a_victim=$(echo "scale=2; ($runtime_for_all_pairs_for_a_victim ) / ($total_orders)" | bc) #runtime_per_victim is runtime for all pairs untill get a pol

        runtime_for_all_victim=$(echo "scale=2; ($avg_runtime_per_order_for_a_victim) + ($runtime_for_all_victim)" | bc) # to get runtime for all the victims

        grep -r ${searching_key_for_prediction_runtime} "$prediction_runtime_filename" > "tmp_prediction_runtime.txt"
        while read pred_line #10 ta order eri
        do
            val=$(echo $pred_line | rev | cut -d',' -f1 | rev)
            total_prediction_runtime=$(echo "scale=2; $total_prediction_runtime + $val" | bc)
        done < "tmp_prediction_runtime.txt"
        rm "tmp_prediction_runtime.txt"

        if [[ $found_in_topN -ge $fifty_percent ]]; then
            vic_count_for_a_module=$((vic_count_for_a_module + 1))
        fi
         
    done < "Org_Only_$type.csv"
    total_victims=$(wc -l < "Org_Only_$type.csv")

    #echo $total_positive_negative_pairs_per_module

    runtime_per_victim_in_a_module=$(echo "scale=2; ($runtime_for_all_victim/$total_victims)" | bc) #after_prediction
    #runtime_per_victim_in_a_module=$(echo "($runtime_per_victim_in_a_module) + ($mvn_overhead)" | bc) #after_prediction
    runtime_per_victim_in_a_module_after_prediction=$(printf "%.2f" $runtime_per_victim_in_a_module)

    #avg_prediction_runtime_in_a_module_during_prediction=$(echo "scale=2; ($total_prediction_runtime)/ ($total_orders * $total_victims)" | bc)
    avg_prediction_runtime_in_a_module_during_prediction=$(echo "scale=2; ($total_prediction_runtime) / ($total_positive_negative_pairs_per_module)" | bc) # ei victim tar jonno total tests kotogulo ache; inference will be based on that; total_positive_negative_pairs_per_module represents all total number of test pairs in a module; For example, if a module has 3 victims, each has 10 orders; Each order contains 100 test-pairs. In this case, I am making total_positive_negative_pairs_per_module=100*10*3; This gives me avg prediction time for a victim
    avg_rank_for_module=$(echo "scale=2; $polluter_rank / ($total_orders * $total_victims) " | bc)
    echo "$row,$vic_count_for_a_module,$avg_rank_for_module,$runtime_per_victim_in_a_module_after_prediction,$avg_prediction_runtime_in_a_module_during_prediction" >> $resultForColumn1

done < $1

