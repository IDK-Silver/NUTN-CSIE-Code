# How to use

## clone repository & move the this folder
```
git clone https://github.com/IDK-Silver/NUTN-CSIE-Code.git
cd ./NUTN-CSIE-Code/Probability/boxplto
```

## install Python packet 
```
pip install -r requirements.txt
```

## run Python script
- the first argument is data path that want to boxplot
- the second argument is result path
```
python boxplot.py ./datas/nuclear_capacities.csv result/nuclear_capacities.png
```

## export result
- if you are running at Unix-like like Linux or MacOS you can easy export result
```
python boxplot.py ./datas/nuclear_capacities.csv result/nuclear_capacities.png >> ./result/result.txt
```  