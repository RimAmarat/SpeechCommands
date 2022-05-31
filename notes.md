
## Preprocessing
### - collate_fn() :
#### pad_sequence() 
```python
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],

-> [[ 711  632   71    0    0    0]
    [  73    8 3215   55  927    0]
    [  83   91    1  645 1253  927]]
```

```python
def collate_fn(batch):
  # For the dataloader that we will need for training
  # for all tensors in the batch
    # get waveform tensors
    # encode targets (label_to_index)

  # pad waveform tensors
  # concatenate target tensors
```