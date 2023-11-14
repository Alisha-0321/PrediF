#!/usr/bin/env bash
if [[ $1 == "" || $2 == "" ]]; then
    echo "arg1 - full path to the test file (eg. tmp.csv)"
    echo "arg2 - "
    exit
fi

currentDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

Result="$currentDir/Result/$2-Tests-Stat.csv"
inputProj="$currentDir/Projects"

echo -n "Module-Name" >> "$Result"
echo -n ",SHA" >> "$Result"
echo -n ",Module" >> "$Result"
echo ",Test-Count" >> "$Result"

while IFS= read -r line
    do
    if [[ ${line} =~ ^\# ]]; then
        echo "Line starts with Hash $line"
        continue
    fi
    slug=$(echo $line | cut -d',' -f1)
    sha=$(echo $line | cut -d',' -f2)
    module=$(echo $line | cut -d',' -f3)
	rootProj=$(echo "$slug" | cut -d/ -f 1)
    subProj=$(echo "$slug" | cut -d/ -f 2)

    if [[ ! -d ${inputProj}/${rootProj} ]]; then
        git clone "https://github.com/$slug" $inputProj/$slug
    fi
    
    cd $inputProj/$slug
    git checkout ${sha}
    if [[ $slug == "zalando/riptide" ]]; then
    	mvn clean install -DskipTests -pl $module -am -Ddependency-check.skip 
    else
    	mvn clean install -DskipTests -pl $module -am
    fi
	#cd $module
    if [[ $module == "dubbo-serialization/dubbo-serialization-fst" ]]; then
	    mvn test -pl $module -pl dubbo-dependencies-bom > log
    else
	    mvn test -pl $module > log
    fi
    total_tests=$(grep -r "Tests run:" "log" | tail -n 1 | cut -d',' -f1 | cut -d':' -f2 | cut -d' ' -f2)
    echo $total_tests
    echo "$slug,$sha,$module,$total_tests" >> "$Result"
	cd $currentDir
    exit
    rm -rf "$inputProj/$rootProj"
done < $1
