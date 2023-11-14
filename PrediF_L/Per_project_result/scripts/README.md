## To get the ranking of each test-pair, run the following command.

```shell
bash runAll.sh data/VP_project_list.csv isVictimPolluterPair
```

```shell
bash runAll.sh data/BSS_project_list.csv isBSSPair
```

```shell
bash runAll.sh data/VC_project_list.csv isVictimPolluterCleanerPair
```
## To get the total test-count of a module, run the following command

```shell
bash find_total_tests.sh data/VP_all_module.txt VP
```

## To get the runtime for each test-pair of VP, BSS, VC, run the following command. But to get the test pass confirm that you clone the https://github.com/TestingResearchIllinois/maven-surefire. Then run this project. Copy surefire-changing-maven-extension/target/surefire-changing-maven-extension-1.0-SNAPSHOT.jar into your Maven installation's /usr/share/maven/lib/ext directory.

```shell
bash run_test_pairs.sh Result/VP_Result.csv VP
```

```shell
bash run_test_pairs.sh Result/BSS_Result.csv BSS
```

```shell
bash run_test_pairs.sh Result/VC_Result.csv VC
```
