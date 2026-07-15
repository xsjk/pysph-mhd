# PySPH MHD

## Installation

```bash
pip install 'pysph[cuda13x] @ git+https://github.com/xsjk/pysph.git'
```

## Cases

```bash
python -m src.cases.alfven -d outputs/alfven/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.jadvect -d outputs/jadvect/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.mhdblast -d outputs/mhdblast/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.mhdrotor -d outputs/mhdrotor/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.mhdshock -d outputs/mhdshock/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.mhdsine -d outputs/mhdsine/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.mhdvortex -d outputs/mhdvortex/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.mhdwave -d outputs/mhdwave/ --cuda --fused --periodic-mode minimum_image
python -m src.cases.orstang -d outputs/orstang/ --cuda --fused --periodic-mode minimum_image
```
