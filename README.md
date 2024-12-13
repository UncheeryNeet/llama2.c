## Character Level Language Model

Based on Llama2.c , I tried to train a character level language model, using enwik8 dataset

## Models

I have trained a small model, and bit larger model


| model | Train Params (M)| Test Params (M) | bpb
| --- | --- | --- | --- | 
TinyLlama base |6|6|1.35
TinyLlama base with parallel final layer|6|6|1.33
TinyLlama large|38|38|1.28
TinyLlama large with parallel final layer|38|38|1.29


## Training

```bash
python enwik8.py download
python enwik8.py pretokenize
```
Then train the model:

```bash
python train_custom.py
python train_custom_parallel.py
```
## Original README
Renamed to README_original.md
