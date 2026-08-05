\# SAT\_RADAR\_AI



\## AI-Driven Weather Radar Emulation Using MTG FCI Satellite Observations



SAT\_RADAR\_AI is a research project developed as part of a PhD study at the South African Weather Service (SAWS). The project investigates the use of Artificial Intelligence, particularly Vision Transformers (ViTs) and Deep Learning, to generate weather radar products from Meteosat Third Generation (MTG) Flexible Combined Imager (FCI) satellite observations.



The long-term objective is to improve precipitation estimation and provide radar-like products in areas where weather radar coverage is unavailable or degraded.



\---



\## Research Objectives



\- Read and process MTG FCI Level-1C satellite data

\- Read SAWS weather radar CF/Radial data

\- Pair satellite and radar observations in space and time

\- Train Vision Transformer models

\- Predict radar reflectivity from satellite observations

\- Evaluate model performance using meteorological verification statistics

\- Generate operational NetCDF radar products



\---



\## Project Structure



```

SAT\_RADAR\_AI/

│

├── data/

│   ├── satellite\_loader.py

│   ├── radar\_loader.py

│   ├── pairing.py

│   └── dataset.py

│

├── models/

├── losses/

├── utils/

├── tests/

│

├── config.py

├── train.py

├── inference.py

├── evaluate.py

│

├── README.md

├── LICENSE

├── requirements.txt

├── environment.yml

└── CITATION.cff

```



\---



\## Current Features



\- MTG FCI Level-1C reader

\- HDF5 JPEG-LS support

\- Satpy integration

\- PyTorch training framework

\- Vision Transformer architecture

\- Radar prediction workflow



\---



\## Planned Features



\- Weather radar CF/Radial reader

\- Lightning integration

\- Solar radiation integration

\- Temporal Transformer

\- Multi-source data fusion

\- Near real-time inference

\- Interactive visualization dashboard



\---



\## Installation



Clone the repository



```bash

git clone https://github.com/Zamikhaya/SAT\_RADAR\_AI.git

cd SAT\_RADAR\_AI

```



Create the Conda environment



```bash

conda env create -f environment.yml

conda activate radar\_SAT\_AI

```



\---



\## Training



```bash

python train.py

```



\---



\## Inference



```bash

python inference.py

```



\---



\## Evaluation



```bash

python evaluate.py

```



\---



\## Author



\*\*Zamikhaya Magogotya\*\*



Specialist: Radar Systems



South African Weather Service



PhD Research in Artificial Intelligence for Weather Radar Applications



\---



\## License



This project is licensed under the MIT License.

