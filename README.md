# PyTorch implementation of RobustFill

Original Paper: https://arxiv.org/pdf/1703.07469.pdf

The RobustFill network by Devlin et al. is trained for the following task -- based on a few example input-output string pairs, generate a program in a domain-specific language that transforms the given inputs into the given outputs.
This program can then be used to transform unseen inputs. For example:

Given these pairs:

| Input              | Output        |
| ------------------ | ------------- |
| Jacob Devlin       | Devlin, J.    |
| Eddy Yeo           | Yeo, E.       |
| Andrej Karpathy    | Karpathy, A.  |
| Anatoly Yakovenko  | Yakovenko, A. |

The RobustFill network will generate a program that can be used to transform an unbounded number of unseen inputs:


| Unseen input      | Transformed Output                        |
| ----------------- | ----------------------------------------- |
| Elon Musk         | <font color="green">Musk, E.</font>       |
| Joe Rogan         | <font color="green">Rogan, J.</font>      |
| Balaji Srinivasan | <font color="green">Srinivasan, B.</font> |

The program generated by our trained network for the example above is as follows:

```python
Concat(
    GetFrom(' '),
    ConstStr(','),
    ConstStr(' '),
    GetSpan(<Type.LOWER: 6>, -4, <Boundary.START: 1>, <Type.WORD: 2>, 5, <Boundary.END: 2>),
    ConstStr('.')
)
```

See the demo notebook to reproduce the result: [Demo Notebook](demo.ipynb)

The network was trained on Google Cloud with 4 x NVIDIA Tesla P4 using PyTorch's `Distributed Data Parallel`.

## Instructions

Set up environment:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Train neural net. The script will automatically use GPU(s) if they are available.

```
python train.py --mode full
```

For testing purposes, run smaller network (on CPU) with a smaller problem size just to see that the loss goes to 0.

```
python train.py --mode easy
```

Run profiler:

```
python train.py --mode profile
```

Run unit tests:

```
python -m unittest
```

Lint:

```
flake8
```
