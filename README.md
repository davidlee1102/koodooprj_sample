----------
<h4 align="center">
   ☀️ ☀️ ☀️ 

Koodoo Project Sample - AI Voice Recognition:


☀️ ☀️ ☀️
</h4>

Hung Le Nhat 


__________
The project will be released in April - 2024

__________
This repository provide the implementation of Koodoo Project Sample and our result that prove our performance

----------

### How to install

```bash
git clone https://github.com/davidlee1102/koodooprj_sample.git
```

### Setup environment

- Install conda software, which can be found at: `https://www.anaconda.com/`
- Import koodooprjsample.yml to conda
- Use this bash

```bash
conda activate koodooprjsample
```

#### OR

```bash
pip install -r requirements.txt
```

Install from requirements

```bash
pip install -r requirements.txt
```

#### OR

Build from docker

```bash
cd KoodooProject
```

```bash
docker-compose build
```

----------
## 🏃‍♂️ Running an Experiment

```bash
python manage.py runserver
```

After server is running, here is some function and the link for request:

- Emotion Recognition(Using Pyannote OR Whisper)

```
http://localhost:8000/emotion_check/
```

```
http://localhost:8000/whisper_emotion_check/
```

- Disclaimer Verification

```
http://localhost:8000/disclaim_check/
```

- Call Summary Generation(Using FalconAI OR BART)

```
http://localhost:8000/conversation_summary/
```
----------
## 🔧‍ Data Backup & Re-Training & Human Feedback

- We provide the pipeline for monitoring and re-training model, using PerfectCloud for managing & re-training. Take a look on `model_retrain.py` file

- The data backup/ logs is saved in csv file `koo_records_logs.csv` file

- The human feedback is provided by using random model for summary, the user will get it from front end.

----------
## 🛠️ Security 

- We provide JSON Web Tokens (JWT) for secure the request. 

## 🔮Future Plans

We have ambitious plans for the future, with a focus on the following priorities:

- **Scale Up:** Our aim is to provide comprehensive recipes and technologies for training massive models on extensive datasets.

..
<h5 align="left">
🤘 This repo is currently developing and fixing, if this repo have any problems or you have any questions/ suggestions, feel free to email us
</h5>
### :smile:If you find this result interesting, please consider to cite this paper:



---
