# steps

1. use the official step to install conda env

2. try to run, but failed.

report error: 
`ModuleNotFoundError: No module named '_swigfaiss_avx2'`,

`ImportError: libcudart.so.10.0: cannot open shared object file: No such file or directory`, 

and

`ModuleNotFoundError: No module named '_swigfaiss'`

then try to install faiss-gpu:
`conda install faiss-gpu -c pytorch`

still failed.

3. try to install dependencies:

(1) `sudo apt install libomp-dev`,  but already installed.

(2) `sudo apt install libopenblas-dev`, but already installed.

(3) `sudo apt install liblapack-dev`, but already installed.

4. try to install faiss-gpu again:

(1) remove:
`conda remove faiss-gpu`

(2) install:

`conda install -c conda-forge faiss-gpu`


got:

```bash
TypeError: TypeErrorDescriptors cannot not be created directly. 
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
```

then:

`pip install protobuf==3.20.0`


Then run the code, got:

```bash
NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
If you want to use the NVIDIA GeForce RTX 3090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```


-----------
# New a conda env and install dependencies manually
since our cuda version is 11.2, 

`conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch` not work.

First search the compatible version for faiss-gpu:1.6.3,

```bash
conda search faiss-gpu=1.6.3 --info -c conda-forge 
```


```bash
conda install faiss-gpu=1.6.3 -c conda-forge
```

try to run the command, got:

```bash

```


Maybe we can 
```
conda install faiss-gpu cudatoolkit=11.0 -c pytorch-gpu
conda install -c anaconda pytorch-gpu
```



----------------------
https://github.com/facebookresearch/faiss/issues/1629:

`conda install -c conda-forge faiss-gpu`

