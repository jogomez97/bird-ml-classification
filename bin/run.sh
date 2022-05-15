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
echo "Run 1/5"
python ${PYTHONPATH}/train_mobilenet.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2'
echo "Run 2/5"
python ${PYTHONPATH}/train_mobilenet.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 41
echo "Run 3/5"
python ${PYTHONPATH}/train_mobilenet.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 1024
echo "Run 4/5"
python ${PYTHONPATH}/train_mobilenet.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 777
echo "Run 5/5"
python ${PYTHONPATH}/train_mobilenet.py -p 'D:/La Salle/Xavier Sevillano Domínguez - TFM Juan Gómez/Dataset espectogrames/birds_spec_224_224_1s_v2' -s 73059

echo "Training done"
