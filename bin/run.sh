#!/bin/bash

if [ -d "../venv/Scripts" ] 
then
    echo "Windows environment." 
    source "..\venv\Scripts\activate"
else
    echo "Linux/MacOs environment"
    source "../venv/bin/activate"
fi

export PYTHONPATH=../src/.

echo "Starting model training"

models=( "mobilenetv2" "resnet50" "vgg16")

for i in "${models[@]}"
do
    echo "Training $i"
    echo "Run 1/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -m $i
    echo "Run 2/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 41 -m $i
    echo "Run 3/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 1024 -m $i
    echo "Run 4/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 777 -m $i
    echo "Run 5/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 73059 -m $i
done


echo "Training done"
