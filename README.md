# Fast yet effective speech emotion recognition with self-distillation

[![arXiv](https://img.shields.io/badge/arXiv-2210.14636-b31b1b.svg)](https://arxiv.org/abs/2210.14636)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### SelfDistill-SER

This is a Python and PyTorch code for the self-distillation framework in our paper: 

>Zhao Ren, Thanh Tam Nguyen, Yi Chang, and Björn W. Schuller. Fast yet effective speech emotion recognition with self-distillation. https://arxiv.org/abs/2210.14636

### Citation

```
@misc{ren2022fast,
      title={Fast yet effective speech emotion recognition with self-distillation}, 
      author={Zhao Ren and Thanh Tam Nguyen and Yi Chang and Björn W. Schuller},
      year={2022},
      eprint={2210.14636},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      note={5 pages}
}
```

## Abstract

In this paper, self-distillation was applied to produce a fast and effective SER model, by simultaneously fine-tuning wav2vec 2.0 and training its shallower versions.

## Config

All of the paths can be set in the runme.sh file. 

## Experiments Running

Preprocessing: main/preprocess.py

Model training: main/main_pytorch.py 

Both python files can be run via

```
sh runme.sh
```

