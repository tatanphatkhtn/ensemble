# ensemble
Run python n-grams.py -u "path/to/unlabel/data" -s "path/to/svr" -n "path/to/non-svr"
-> output result to ng-result

Run python dataPreProcessing.py -f <frequency (filter out attribute by frequency default: 50%)>

-> output to pp-result and pp-result/filtered

-> use result from pp-result/filtered

ex cmd: python main.py -u ./pp-result/filtered/filteredUnlMatrix0.3.npy -l ./pp-result/filtered/filteredLalMatrix0.3.npy -t ./ng-result/target.txt -a ./pp-result/filtered/filteredAttr0.3.npy -e 4


Usage: python main.py -l "path/to/lal/data" -u "path/to/unlal/data" -t "path/to/target/array" -a "path/to/attr/array" -v "Log mode" -e "number of estimators"

Use: main.py --help for more info

Output: Graph with 2 sub figures:

First one: Blue line illutrates for only semi run, red line illutrates for semi + sffs run (by accuracy and number of classifiers)
Second one: Nevermind :) 
