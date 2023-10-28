#!/bin/bash
if [[ $1 == "" || $2 == "" ]]; then
    echo "Give the input file that saves your tests name (Result/VP_Result.csv)" 
    echo "Give the type (VP/VC/BSS)" 
    exit
fi

currentDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
inputProj=$currentDir"/Projects"
Result="$currentDir/Result"
Log="$currentDir/logs"
if [[ ! -d $Result ]]; then
    mkdir $Result
fi


if [[ ! -d $inputProj ]]; then
    mkdir $inputProj
fi

if [ ! -d "$Log" ]; then
    mkdir "$Log"
fi

echo -n "Slug" >> "$Result/$2-Test-Pair-Runtime.csv"
echo -n ",SHA" >> "$Result/$2-Test-Pair-Runtime.csv"
echo -n ",Module" >> "$Result/$2-Test-Pair-Runtime.csv"
echo ",Runtime" >> "$Result/$2-Test-Pair-Runtime.csv"

MVNOPTIONS="-Dsurefire.runOrder=testOrder"

row_count=0
while read line
do
    JMVNOPTIONS="$MVNOPTIONS"
    MVNINSTALLOPTIONS=""
    if [[ ${line} =~ ^\# ]]; then
        echo "Line starts with Hash $line"
        continue
    fi 

    row_count=$((row_count + 1)) 
    if [[ $row_count -eq 1 ]]; then # To ignore the header
        continue
    fi

	slug=$(echo ${line} | cut -d',' -f1)
    if [[ $slug == "AVG" ]]; then
        continue
    fi
    modified_slug=$(echo "$slug" | sed 's/\//_/')
    rootProj=$(echo "$slug" | cut -d/ -f 1)
    subProj=$(echo "$slug" | cut -d/ -f 2)
    
    if [[ $2 == "VC" ]]; then 
        test_pair_with_dot=$(echo $line | rev | cut -d',' -f1-3 | rev)
        cleaner=$(echo $line | rev | cut -d',' -f1 | sed 's/\./#/' | rev )
        polluter=$(echo $line | rev | cut -d',' -f2 | sed 's/\./#/' | rev )
        victim=$(echo $line | rev | cut -d',' -f3  | sed 's/\./#/' | rev)
        test_pair="$polluter,$cleaner,$victim"
        echo $test_pair
    elif [[ $2 == "VP" ]]; then
        test_pair_with_dot=$(echo $line | rev | cut -d',' -f1-2 | rev)
        polluter=$(echo $line | rev | cut -d',' -f1 | sed 's/\./#/' | rev )
        victim=$(echo $line | rev | cut -d',' -f2 | sed 's/\./#/' | rev )
        test_pair="$polluter,$victim"
        echo $test_pair
    elif [[ $2 == "BSS" ]]; then
        test_pair_with_dot=$(echo $line | rev | cut -d',' -f1-2 | rev)
        state_setter=$(echo $line | rev | cut -d',' -f1 | sed 's/\./#/' | rev )
        brittle=$(echo $line | rev | cut -d',' -f2 | sed 's/\./#/' | rev )
        test_pair="$state_setter,$brittle"
    fi

    dir="../../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_Comments"
    echo $test_pair_with_dot
    echo "${dir}/${2}/${2}_${modified_slug}.csv"
    grep_result=$(grep -r "$test_pair_with_dot" "${dir}/${2}/${2}_${modified_slug}.csv" | head -n 1 | cut -d',' -f1-3) # Will contain slug,sha,module
    echo $grep_result
    echo $test_pair_with_dot
	sha=$(echo ${grep_result} | cut -d',' -f2)
	module=$(echo ${grep_result} | cut -d',' -f3)
    echo $module
    #echo $modified_module

    if [[ ! -d ${inputProj}/${rootProj} ]]; then
        git clone "https://github.com/$slug" $inputProj/$slug
    fi

    cd $inputProj/$slug

    git checkout $sha
    echo $module
    #exit
    if [[ $module == "" ]]; then
        echo "module empty"
        module="."
    fi
    echo $sha
    if [[ $slug == "zalando/riptide" ]]; then
    	#mvn clean install -DskipTests -pl $module -am -Ddependency-check.skip 
        MVNINSTALLOPTIONS="-Ddependency-check.skip=true"
    elif [[ "$slug" == "apache/hadoop" ]]; then
        sudo apt-get install autoconf automake libtool curl make g++ unzip -y --allow-unauthenticated;
        wget -nv https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz;
        tar -zxvf protobuf-2.5.0.tar.gz;
        cd protobuf-2.5.0
        ./configure; make -j15;
        su shanto-admin
        sudo make install;
        sudo ldconfig;
        cd ..
    elif [[ $slug == "dropwizard/dropwizard" ]]; then
    	#mvn clean install -DskipTests -pl $module -am -Ddependency-check.skip=true
        MVNINSTALLOPTIONS="-Ddependency-check.skip=true"
        #echo "mvn clean install -DskipTests -pl $module ${MVNOPTIONS} -am"
	elif [[ "$slug" == "openpojo/openpojo" ]]; then
    	sed -i '70s/.*/return null;/' src/main/java/com/openpojo/random/generator/security/CredentialsRandomGenerator.java
    elif [[ "$slug" == "doanduyhai/Achilles" ]]; then
        sed -i 's~http://repo1.maven.org/maven2~https://repo1.maven.org/maven2~g' pom.xml
    elif [[ "$slug" == "spring-projects/spring-data-envers" ]]; then
        sed -i 's~2.2.0.BUILD-SNAPSHOT~2.2.0.RELEASE~g' pom.xml
    fi
   
    mvn clean install -DskipTests -pl $module -am ${MVNINSTALLOPTIONS}


    start_time=$(date +%s.%N) 
    if [[ $slug == "apache/incubator-dubbo" ]]; then 
        JMVNOPTIONS="${MVNOPTIONS} -pl dubbo-dependencies-bom"
    elif [[ $slug == "dropwizard/dropwizard" ]]; then
        echo "TESTSING*********************************************"
        JMVNOPTIONS="${MVNOPTIONS} -pl dropwizard-bom/"
    fi
    mvn test ${JMVNOPTIONS}  -pl ${module} -Dtest=${test_pair} |& tee "$Log/${modified_slug}_$2_${row_count}.csv"
    end_time=$(date +%s.%N) 
    echo $end_time
    take=$(echo "scale=2; ${end_time} - ${start_time}" | bc)

    #Parsing surefire
    pip install BeautifulSoup4
    pip install lxml
    
    echo "===========$Log/${modified_slug}_$2_${row_count}.csv ===========" >> "$currentDir/$2_test-results.csv"
    for testname in $(egrep "Running " "$Log/${modified_slug}_$2_${row_count}.csv" | rev | cut -d' ' -f1 | rev); do
       echo "g=$testname"
       echo "TEST-${testname}.xml"
       testname_clean=$(echo "$testname" | sed -r 's/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]//g')
       f=$(find -name "TEST-${testname_clean}.xml") 
       python3 $currentDir/parse_surefire_report.py $f 1 $testname  >> "$currentDir/$2_test-results.csv"
    done
    
    
    echo "$slug,$sha,$module,${test_pair},$take" >> "$Result/$2-Test-Pair-Runtime.csv" 
    cd $currentDir
<<<<<<< Updated upstream
    #rm -rf "$inputProj/$rootProj"
=======
    exit
    rm -rf "$inputProj/$rootProj"
>>>>>>> Stashed changes
done < $1
