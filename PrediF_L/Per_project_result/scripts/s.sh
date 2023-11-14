#bash runAll.sh data/VC_all_module.txt isVictimPolluterCleanerPair 1
#bash runAll.sh data/x isVictimPolluterCleanerPair 1
#bash runAll.sh data/BSS_all_module.txt isBSSPair 1
bash runAll.sh data/VP_all_module.txt isVictimPolluterPair 1
#bash runAll.sh data/x isVictimPolluterPair 1

#Once running the command for VP and BSS, then I will search for each victim. If we find one polluter is ranked at top 1 for atleast one order, then that victim, we will consider with polluter in top 1.

#grep -r ",1,Class" Result/VP_On_10_Failing_Order_Result.csv > l
#grep -r ",2,Class" Result/VP_On_10_Failing_Order_Result.csv >> l
#grep -r ",3,Class" Result/VP_On_10_Failing_Order_Result.csv >> l
#cut -d',' -f1-4 l > tmp

#sort tmp | uniq > unique_lines.txt
#declare -A mymap
#
#while read line 
#do
#    line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
#    count=$(grep -r "$line" tmp | wc -l)
#    if [[ $count -ge 5 ]]; then #Expecting at least 50% of the test-orders, we find atleast one polluter
#        mymap["$line"]=$count
#    fi
#done < "unique_lines.txt"
#
## Printing all keys and values
#for key in "${!mymap[@]}"; do
#    echo "$key: ${mymap[$key]}"
#done
