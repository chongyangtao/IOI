# IOI

This is an implementation of [One Time of Interaction May Not Be Enough: Go Deep with an Interaction-over-Interaction Network for Response Selection in Dialogues, ACL 2019].


## Requirements
* Ubuntu 16.04
* Tensorflow 1.4.0
* Python 3.5


## Usage
Prepare a pre-trained word2vec file and preprocess the data, run

```bash
# preprocess the data
python data_utils_record.py
```

All hyper parameters are stored in config.py. To train, run

```bash
python main.py --log_root=logs_ubuntu --batch_size=20
```

To evaluate the model, run
```bash
python evaluate.py --log_root=logs_ubuntu --batch_size=20
```
