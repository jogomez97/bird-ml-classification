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

    batch=32
    if [ $i = "vgg16" ]; then
        batch=16
    fi

    echo "Run 1/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -m $i -b $batch
    echo "Run 2/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 42 -m $i -b $batch
    echo "Run 3/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 1024 -m $i -b $batch
    echo "Run 4/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 777 -m $i -b $batch
    echo "Run 5/5 of $i"
    python ${PYTHONPATH}/train.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 73059 -m $i -b $batch
done


echo "Training done"
