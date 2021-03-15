# How to compile HTK under CUDA

This document is mostly directed towards WSL2, however it can be easily used for Linux based compilation.

# Preliminaries

You can compile HTK with CUDA support either on Windows or on Linux:

## WSL2

Windows Services for Linux 2 (WSL2). In order to compile HTK using CUDA Toolkit for Windows you need to change your Windows version to Microsoft Insiders Program.
Below you can find more details on instructions how to enable :
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Below you can find the version that I have on my machine:
``` 
$ /usr/local/cuda/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Wed_Jul_22_19:09:09_PDT_2020
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0
```

## Linux

Download and install CUDA Toolkit (assuming you have a CUDA driver installed properly on your Linux).
https://developer.nvidia.com/cuda-downloads?target_os=Linux

# Installation of CUDNN library

Download CUDNN
https://developer.nvidia.com/rdp/cudnn-download
Choose 64 bit version for Linux from:
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz

a) Unpack include files and manually copy them to: 
```
/usr/local/cuda-11/include
```
or other location depending where your CUDA Toolkit was installed. In my case on WSL2 the CUDA Toolkit was installed in `/usr/local/cuda-11`.

The list of files to copy (on Windows WSL2):
```
> dir
15.03.2021  10:17    <DIR>          .
15.03.2021  10:17    <DIR>          ..
25.02.2021  09:18             2 968 cudnn.h
25.02.2021  09:18            29 025 cudnn_adv_infer.h
25.02.2021  09:18            29 025 cudnn_adv_infer_v8.h
25.02.2021  09:18            27 700 cudnn_adv_train.h
25.02.2021  09:18            27 700 cudnn_adv_train_v8.h
25.02.2021  09:18            15 708 cudnn_backend.h
25.02.2021  09:18            15 708 cudnn_backend_v8.h
25.02.2021  09:18            29 011 cudnn_cnn_infer.h
25.02.2021  09:18            29 011 cudnn_cnn_infer_v8.h
25.02.2021  09:18            10 177 cudnn_cnn_train.h
25.02.2021  09:18            10 177 cudnn_cnn_train_v8.h
25.02.2021  09:18            48 968 cudnn_ops_infer.h
25.02.2021  09:18            48 968 cudnn_ops_infer_v8.h
25.02.2021  09:18            25 733 cudnn_ops_train.h
25.02.2021  09:18            25 733 cudnn_ops_train_v8.h
25.02.2021  09:18             2 968 cudnn_v8.h
25.02.2021  09:18             2 785 cudnn_version.h
25.02.2021  09:18             2 785 cudnn_version_v8.h
```

b) Unpack library files and manuall copy them to: 
```
/usr/local/cuda-11/lib64
```
or other location depending where your CUDA Toolkit was installed. In my case on WSL2 the CUDA Toolkit was installed in `/usr/local/cuda-11`.

The list of files to copy (on Windows WSL2):
```
25.02.2021  09:18           158 264 libcudnn.so
25.02.2021  09:18           158 264 libcudnn.so.8
25.02.2021  09:18           158 264 libcudnn.so.8.1.1
25.02.2021  09:18       127 363 056 libcudnn_adv_infer.so
25.02.2021  09:18       127 363 056 libcudnn_adv_infer.so.8
25.02.2021  09:18       127 363 056 libcudnn_adv_infer.so.8.1.1
25.02.2021  09:18        82 375 312 libcudnn_adv_train.so
25.02.2021  09:18        82 375 312 libcudnn_adv_train.so.8
25.02.2021  09:18        82 375 312 libcudnn_adv_train.so.8.1.1
25.02.2021  09:18       672 420 792 libcudnn_cnn_infer.so
25.02.2021  09:18       672 420 792 libcudnn_cnn_infer.so.8
25.02.2021  09:18       672 420 792 libcudnn_cnn_infer.so.8.1.1
25.02.2021  09:18        99 677 120 libcudnn_cnn_train.so
25.02.2021  09:18        99 677 120 libcudnn_cnn_train.so.8
25.02.2021  09:18        99 677 120 libcudnn_cnn_train.so.8.1.1
25.02.2021  09:18       284 239 304 libcudnn_ops_infer.so
25.02.2021  09:18       284 239 304 libcudnn_ops_infer.so.8
25.02.2021  09:18       284 239 304 libcudnn_ops_infer.so.8.1.1
25.02.2021  09:18        46 338 264 libcudnn_ops_train.so
25.02.2021  09:18        46 338 264 libcudnn_ops_train.so.8
25.02.2021  09:18        46 338 264 libcudnn_ops_train.so.8.1.1
25.02.2021  09:18     1 507 075 052 libcudnn_static.a
25.02.2021  09:18     1 507 075 052 libcudnn_static_v8.a
```

Similar procedure applies to Linux (not only WSL2), but in case of usual Linux you need to find a correct version of CUDNN library and after downloading, unpacking copy the include and lib files to the location specified above.

# Compilation of HTKLib

1. Go to HTKLib:
```
$ cd HTKLib:
```

2. Prepare compilation as follows:

Modify MakefileNVCC and replace any occurences of:

a) 'compute_75' to the version you might be using 'compute_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).

b) 'sm_75' to the version you might be using 'sm_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).
 
3. Compile HTKLib as follows:

a) First clean up:
```
HTKLib$ make -f MakefileNVCC clean
rm -f HGraf.o esig_asc.o esig_edr.o esignal.o esig_nat.o HAdapt.o HANNet.o HArc.o HAudio.o HCUDA.o HDict.o HExactMPE.o HFB.o HFBLat.o HLabel.o HLat.o HLM.o HNLM.o HRNLM.o HMap.o HMath.o HMem.o HModel.o HNCache.o HNet.o HParm.o HRec.o HShell.o HSigP.o HTrain.o HUtil.o HVQ.o HWave.o HGraf.lv.o esig_asc.lv.o esig_edr.lv.o esignal.lv.o esig_nat.lv.o HAdapt.lv.o HANNet.lv.o HArc.lv.o HAudio.lv.o HCUDA.lv.o HDict.lv.o HExactMPE.lv.o HFB.lv.o HFBLat.lv.o HLabel.lv.o HLat.lv.o HLM.lv.o HMap.lv.o HMath.lv.o HMem.lv.o HModel.lv.o HNCache.lv.o HNet.lv.o HParm.lv.o HRec.lv.o HShell.lv.o HSigP.lv.o HTrain.lv.o HUtil.lv.o HVQ.lv.o HWave.lv.o  HTKLib.a HTKLiblv.a
```

b) Then compile:
```
HTKLib$ make -f MakefileNVCC all
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HGraf.o HGraf.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o esig_asc.o esig_asc.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o esig_edr.o esig_edr.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o esignal.o esignal.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o esig_nat.o esig_nat.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HAdapt.o HAdapt.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HANNet.o HANNet.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HArc.o HArc.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HAudio.o HAudio.c
/usr/local/cuda/bin/nvcc -o HCUDA.o -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c HCUDA.cu
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HDict.o HDict.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HExactMPE.o HExactMPE.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HFB.o HFB.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HFBLat.o HFBLat.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HLabel.o HLabel.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HLat.o HLat.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HLM.o HLM.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HNLM.o HNLM.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HRNLM.o HRNLM.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HMap.o HMap.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HMath.o HMath.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HMem.o HMem.c
HMem.c: In function ‘CreateNVector’:
HMem.c:1134:5: warning: implicit declaration of function ‘DevNew’ [-Wimplicit-function-declaration]
 1134 |     DevNew(&v->devElems, NVectorElemSize(nlen));
      |     ^~~~~~
HMem.c: In function ‘FreeNVector’:
HMem.c:1252:9: warning: implicit declaration of function ‘DevDispose’; did you mean ‘Dispose’? [-Wimplicit-function-declaration]
 1252 |         DevDispose(v->devElems, NVectorElemSize(v->vecLen));
      |         ^~~~~~~~~~
      |         Dispose
HMem.c: In function ‘SyncNVectorDev2Host’:
HMem.c:1291:5: warning: implicit declaration of function ‘SyncDev2Host’; did you mean ‘SyncNMatrixDev2Host’? [-Wimplicit-function-declaration]
 1291 |     SyncDev2Host(v->devElems, v->vecElems, NVectorElemSize(v->vecLen));
      |     ^~~~~~~~~~~~
      |     SyncNMatrixDev2Host
HMem.c: In function ‘SyncNVectorHost2Dev’:
HMem.c:1298:5: warning: implicit declaration of function ‘SyncHost2Dev’; did you mean ‘SyncNMatrixHost2Dev’? [-Wimplicit-function-declaration]
 1298 |     SyncHost2Dev(v->vecElems, v->devElems, NVectorElemSize(v->vecLen));
      |     ^~~~~~~~~~~~
      |     SyncNMatrixHost2Dev
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HModel.o HModel.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HNCache.o HNCache.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HNet.o HNet.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HParm.o HParm.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HRec.o HRec.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HShell.o HShell.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HSigP.o HSigP.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HTrain.o HTrain.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HUtil.o HUtil.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HVQ.o HVQ.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA   -c -o HWave.o HWave.c
if [ -f HTKLib.a ] ; then  /bin/rm HTKLib.a ; fi
ar rv HTKLib.a HGraf.o esig_asc.o esig_edr.o esignal.o esig_nat.o HAdapt.o HANNet.o HArc.o HAudio.o HCUDA.o HDict.o HExactMPE.o HFB.o HFBLat.o HLabel.o HLat.o HLM.o HNLM.o HRNLM.o HMap.o HMath.o HMem.o HModel.o HNCache.o HNet.o HParm.o HRec.o HShell.o HSigP.o HTrain.o HUtil.o HVQ.o HWave.o
ar: creating HTKLib.a
a - HGraf.o
a - esig_asc.o
a - esig_edr.o
a - esignal.o
a - esig_nat.o
a - HAdapt.o
a - HANNet.o
a - HArc.o
a - HAudio.o
a - HCUDA.o
a - HDict.o
a - HExactMPE.o
a - HFB.o
a - HFBLat.o
a - HLabel.o
a - HLat.o
a - HLM.o
a - HNLM.o
a - HRNLM.o
a - HMap.o
a - HMath.o
a - HMem.o
a - HModel.o
a - HNCache.o
a - HNet.o
a - HParm.o
a - HRec.o
a - HShell.o
a - HSigP.o
a - HTrain.o
a - HUtil.o
a - HVQ.o
a - HWave.o
ranlib HTKLib.a
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HGraf.lv.o HGraf.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o esig_asc.lv.o esig_asc.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o esig_edr.lv.o esig_edr.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o esignal.lv.o esignal.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o esig_nat.lv.o esig_nat.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HAdapt.lv.o HAdapt.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HANNet.lv.o HANNet.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HArc.lv.o HArc.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HAudio.lv.o HAudio.c
/usr/local/cuda/bin/nvcc -o HCUDA.lv.o -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c HCUDA.cu
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HDict.lv.o HDict.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HExactMPE.lv.o HExactMPE.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HFB.lv.o HFB.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HFBLat.lv.o HFBLat.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HLabel.lv.o HLabel.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HLat.lv.o HLat.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HLM.lv.o HLM.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HMap.lv.o HMap.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HMath.lv.o HMath.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HMem.lv.o HMem.c
HMem.c: In function ‘CreateNVector’:
HMem.c:1134:5: warning: implicit declaration of function ‘DevNew’ [-Wimplicit-function-declaration]
 1134 |     DevNew(&v->devElems, NVectorElemSize(nlen));
      |     ^~~~~~
HMem.c: In function ‘FreeNVector’:
HMem.c:1252:9: warning: implicit declaration of function ‘DevDispose’; did you mean ‘Dispose’? [-Wimplicit-function-declaration]
 1252 |         DevDispose(v->devElems, NVectorElemSize(v->vecLen));
      |         ^~~~~~~~~~
      |         Dispose
HMem.c: In function ‘SyncNVectorDev2Host’:
HMem.c:1291:5: warning: implicit declaration of function ‘SyncDev2Host’; did you mean ‘SyncNMatrixDev2Host’? [-Wimplicit-function-declaration]
 1291 |     SyncDev2Host(v->devElems, v->vecElems, NVectorElemSize(v->vecLen));
      |     ^~~~~~~~~~~~
      |     SyncNMatrixDev2Host
HMem.c: In function ‘SyncNVectorHost2Dev’:
HMem.c:1298:5: warning: implicit declaration of function ‘SyncHost2Dev’; did you mean ‘SyncNMatrixHost2Dev’? [-Wimplicit-function-declaration]
 1298 |     SyncHost2Dev(v->vecElems, v->devElems, NVectorElemSize(v->vecLen));
      |     ^~~~~~~~~~~~
      |     SyncNMatrixHost2Dev
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HModel.lv.o HModel.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HNCache.lv.o HNCache.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HNet.lv.o HNet.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HParm.lv.o HParm.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HRec.lv.o HRec.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HShell.lv.o HShell.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HSigP.lv.o HSigP.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HTrain.lv.o HTrain.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HUtil.lv.o HUtil.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HVQ.lv.o HVQ.c
/usr/local/cuda/bin/nvcc -DNO_LAT_LM -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -c -o HWave.lv.o HWave.c
if [ -f HTKLiblv.a ] ; then  /bin/rm HTKLiblv.a ; fi
ar rv HTKLiblv.a HGraf.lv.o esig_asc.lv.o esig_edr.lv.o esignal.lv.o esig_nat.lv.o HAdapt.lv.o HANNet.lv.o HArc.lv.o HAudio.lv.o HCUDA.lv.o HDict.lv.o HExactMPE.lv.o HFB.lv.o HFBLat.lv.o HLabel.lv.o HLat.lv.o HLM.lv.o HMap.lv.o HMath.lv.o HMem.lv.o HModel.lv.o HNCache.lv.o HNet.lv.o HParm.lv.o HRec.lv.o HShell.lv.o HSigP.lv.o HTrain.lv.o HUtil.lv.o HVQ.lv.o HWave.lv.o
ar: creating HTKLiblv.a
a - HGraf.lv.o
a - esig_asc.lv.o
a - esig_edr.lv.o
a - esignal.lv.o
a - esig_nat.lv.o
a - HAdapt.lv.o
a - HANNet.lv.o
a - HArc.lv.o
a - HAudio.lv.o
a - HCUDA.lv.o
a - HDict.lv.o
a - HExactMPE.lv.o
a - HFB.lv.o
a - HFBLat.lv.o
a - HLabel.lv.o
a - HLat.lv.o
a - HLM.lv.o
a - HMap.lv.o
a - HMath.lv.o
a - HMem.lv.o
a - HModel.lv.o
a - HNCache.lv.o
a - HNet.lv.o
a - HParm.lv.o
a - HRec.lv.o
a - HShell.lv.o
a - HSigP.lv.o
a - HTrain.lv.o
a - HUtil.lv.o
a - HVQ.lv.o
a - HWave.lv.o
ranlib HTKLiblv.a
```

c) The directory contents should look as follows:
```
HTKLib$ ls
 ls
HANNet.c     HCUDA.cu        HFBLat.c      HLabel.lv.o  HMem.lv.o     HNet.o       HSigP.c      HVQ.lv.o       esig_nat.c
HANNet.h     HCUDA.h         HFBLat.h      HLabel.o     HMem.o        HParm.c      HSigP.h      HVQ.o          esig_nat.lv.o
HANNet.lv.o  HCUDA.lv.o      HFBLat.lv.o   HLat.c       HModel.c      HParm.h      HSigP.lv.o   HWave.c        esig_nat.o
HANNet.o     HCUDA.o         HFBLat.o      HLat.h       HModel.h      HParm.lv.o   HSigP.o      HWave.h        esignal.c
HAdapt.c     HDict.c         HGraf.c       HLat.lv.o    HModel.lv.o   HParm.o      HTKLib.a     HWave.lv.o     esignal.h
HAdapt.h     HDict.h         HGraf.h       HLat.o       HModel.o      HRNLM.c      HTKLiblv.a   HWave.o        esignal.lv.o
HAdapt.lv.o  HDict.lv.o      HGraf.lv.o    HMap.c       HNCache.c     HRNLM.h      HTrain.c     MakefileCPU    esignal.o
HAdapt.o     HDict.o         HGraf.null.c  HMap.h       HNCache.h     HRNLM.o      HTrain.h     MakefileMKL    lib
HArc.c       HExactMPE.c     HGraf.o       HMap.lv.o    HNCache.lv.o  HRec.c       HTrain.lv.o  MakefileNVCC
HArc.h       HExactMPE.h     HLM-RNNLM.c   HMap.o       HNCache.o     HRec.h       HTrain.o     config.h
HArc.lv.o    HExactMPE.lv.o  HLM.c         HMath.c      HNLM.c        HRec.lv.o    HUtil.c      esig_asc.c
HArc.o       HExactMPE.o     HLM.h         HMath.h      HNLM.h        HRec.o       HUtil.h      esig_asc.lv.o
HAudio.c     HFB.c           HLM.lv.o      HMath.lv.o   HNLM.o        HShell.c     HUtil.lv.o   esig_asc.o
HAudio.h     HFB.h           HLM.o         HMath.o      HNet.c        HShell.h     HUtil.o      esig_edr.c
HAudio.lv.o  HFB.lv.o        HLabel.c      HMem.c       HNet.h        HShell.lv.o  HVQ.c        esig_edr.lv.o
HAudio.o     HFB.o           HLabel.h      HMem.h       HNet.lv.o     HShell.o     HVQ.h        esig_edr.o
```

# Compilation of HTKTools

1. Go to HTKTools:
```
HTKLib$ cd ../HTKTools
```

2. Prepare compilation as follows:

Modify MakefileNVCC and replace any occurences of:

a) 'compute_75' to the version you might be using 'compute_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).

b) 'sm_75' to the version you might be using 'sm_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).
 
3. Compile HTKTools as follows:

a) First clean up:
```
HTKTools$ make -f MakefileNVCC clean
rm -f *.o
```

b) Then compile:
```
HTKTools$ make -f MakefileNVCC all
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHBuild = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HBuild -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HBuild.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HBuild -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HBuild.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HBuild ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHCompV = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HCompV -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HCompV.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HCompV -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HCompV.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HCompV ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHCopy = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HCopy -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HCopy.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HCopy -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HCopy.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HCopy ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHDMan = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HDMan -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HDMan.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HDMan -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HDMan.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HDMan ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHERest = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HERest -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HERest.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HERest -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HERest.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HERest ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHHEd = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HHEd -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HHEd.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HHEd -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HHEd.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
HHEd.c: In function ‘main’:
HHEd.c:287:4: warning: implicit declaration of function ‘InitCUDA’ [-Wimplicit-function-declaration]
  287 |    InitCUDA();
      |    ^~~~~~~~
HHEd.c: In function ‘DuplicateCommand’:
HHEd.c:4177:35: warning: ‘%s’ directive writing up to 254 bytes into a region of size between 1 and 255 [-Wformat-overflow=]
 4177 |                   sprintf(name,"%s%s+%s",buf,id,p);
      |                                   ^~         ~~
HHEd.c:4177:19: note: ‘sprintf’ output 2 or more bytes (assuming 510) into a destination of size 255
 4177 |                   sprintf(name,"%s%s+%s",buf,id,p);
      |                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HHEd.c: In function ‘MakeIntoMacrosCommand’:
HHEd.c:4561:27: warning: ‘sprintf’ may write a terminating nul past the end of the destination [-Wformat-overflow=]
 4561 |          sprintf(buf,"%s%d",macName,++n);
      |                           ^
HHEd.c:4561:10: note: ‘sprintf’ output between 2 and 266 bytes into a destination of size 255
 4561 |          sprintf(buf,"%s%d",macName,++n);
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HHEd.c: In function ‘CreateTMRecs’:
HHEd.c:5169:20: warning: ‘%s’ directive writing up to 2047 bytes into a region of size 80 [-Wformat-overflow=]
 5169 |       sprintf(buf,"%s_%d_",tiedMixName,s);
      |                    ^~      ~~~~~~~~~~~
HHEd.c:5169:7: note: ‘sprintf’ output between 4 and 2061 bytes into a destination of size 80
 5169 |       sprintf(buf,"%s_%d_",tiedMixName,s);
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HHEd ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHInit = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HInit -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HInit.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HInit -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HInit.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HInit ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHLEd = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HLEd -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLEd.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HLEd -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLEd.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HLEd ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHList = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HList -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HList.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HList -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HList.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HList ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHLConf = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HLConf -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLConf.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HLConf -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLConf.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HLConf ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHLRescore = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HLRescore -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLRescore.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HLRescore -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLRescore.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HLRescore ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHLStats = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HLStats -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLStats.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HLStats -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HLStats.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HLStats ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHMMIRest = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HMMIRest -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HMMIRest.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HMMIRest -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HMMIRest.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HMMIRest ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHNTrainSGD = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HNTrainSGD -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HNTrainSGD.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HNTrainSGD -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HNTrainSGD.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HNTrainSGD ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHNForward = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HNForward -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HNForward.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HNForward -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HNForward.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HNForward ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHParse = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HParse -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HParse.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HParse -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HParse.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HParse ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHQuant = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HQuant -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HQuant.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HQuant -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HQuant.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HQuant ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHRest = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HRest -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HRest.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HRest -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HRest.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HRest ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHResults = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HResults -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HResults.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HResults -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HResults.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HResults ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHSGen = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HSGen -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HSGen.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HSGen -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HSGen.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HSGen ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHSmooth = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HSmooth -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HSmooth.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HSmooth -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HSmooth.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HSmooth ../bin.gpu  ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
if [ xHVite = xHSLab ] ; then \
        /usr/local/cuda/bin/nvcc -o HVite -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HVite.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm -lX11 ; \
        else \
        /usr/local/cuda/bin/nvcc -o HVite -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -I../HTKLib  HVite.c ../HTKLib/HTKLib.a -L/usr/X11R6/lib -lcudart -lcublas -lcurand -lcudnn -lm ; fi
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HVite ../bin.gpu  ; fi
```

c) The directory contents should look as follows:
```
HTKTools$ ls
HBuild    HCopy.c   HHEd     HLConf.c     HLStats    HMMIRest.c    HParse    HRest.c     HSLab.c    MakefileCPU
HBuild.c  HDMan     HHEd.c   HLEd         HLStats.c  HNForward     HParse.c  HResults    HSmooth    MakefileMKL
HCompV    HDMan.c   HInit    HLEd.c       HList      HNForward.c   HQuant    HResults.c  HSmooth.c  MakefileNVCC
HCompV.c  HERest    HInit.c  HLRescore    HList.c    HNTrainSGD    HQuant.c  HSGen       HVite
HCopy     HERest.c  HLConf   HLRescore.c  HMMIRest   HNTrainSGD.c  HRest     HSGen.c     HVite.c
```

d) Test if HTKTools are successfully compiled by issuing:
```
HTKTools$ ./HVite

USAGE: HVite [options] VocabFile HMMList DataFiles...

 Option                                       Default

 -a      align from label files               off
 -b s    def s as utterance boundary word     none
 -c f    tied mixture pruning threshold       10.0
 -d s    dir to find hmm definitions          current
 -e      save direct audio rec output         off
 -f      output full state alignment          off
 -g      enable audio replay                  off
 -h s    set speaker name pattern             *.mfc
 -i s    Output transcriptions to MLF s       off
 -j i    Online MLLR adaptation               off
         Perform update every i utterances
 -k      use an input transform               off
 -l s    dir to store label/lattice files     current
 -m      output model alignment               off
 -n i [N] N-best recognition (using i tokens) off
 -o s    output label formating NCSTWMX       none
 -p f    inter model trans penalty (log)      0.0
 -q s    output lattice formating ABtvaldmn   tvaldmn
 -r f    pronunciation prob scale factor      1.0
 -s f    grammar scale factor                 1.0
 -t f [f f] set pruning threshold             0.0
 -u i    set pruning max active               0
 -v f    set word end pruning threshold       0.0
 -w [s]  recognise from network               off
 -x s    extension for hmm files              none
 -y s    output label file extension          rec
 -z s    generate lattices with extension s   off
 -A      Print command line arguments         off
 -B      Save HMMs/transforms as binary       off
 -C cf   Set config file to cf                default
 -D      Display configuration variables      off
 -E s [s] set dir for parent xform to s       off
         and optional extension
 -F fmt  Set source data format to fmt        as config
 -G fmt  Set source label format to fmt       as config
 -H mmf  Load HMM macro file mmf
 -I mlf  Load master label file mlf
 -J s [s] set dir for input xform to s        none
         and optional extension
 -K s [s] set dir for output xform to s       none
         and optional extension
 -L dir  Set input label (or net) dir         current
 -P      Set target label format to fmt       as config
 -S f    Set script file to f                 none
 -T N    Set trace flags to N                 0
 -V      Print version information            off
 -X ext  Set input label (or net) file ext    lab 
```

A sample version output:
```
HTKTools$  ./HVite -V

HTK Version Information
Module     Version    Who    Date      : CVS Info
HVite      3.5.0      CUED   12/10/15  : $Id: HVite.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HShell     3.5.0      CUED   12/10/15  : $Id: HShell.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HMem       3.5.0      CUED   12/10/15  : $Id: HMem.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HLabel     3.5.0      CUED   12/10/15  : $Id: HLabel.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HMath      3.5.0      CUED   12/10/15  : $Id: HMath.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HSigP      3.5.0      CUED   12/10/15  : $Id: HSigP.c,v 1.1.1.1 2006/10/11 09:54:58 jal58 Exp $
HWave      3.5.0      CUED   12/10/15  : $Id: HWave.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HAudio     3.5.0      CUED   12/10/15  : $Id: HAudio.c,v 1.1.1.1 2006/10/11 09:54:57 jal58 Exp $
HVQ        3.5.0      CUED   12/10/15  : $Id: HVQ.c,v 1.1.1.1 2006/10/11 09:54:59 jal58 Exp $
HModel     3.5.0      CUED   12/10/15  : $Id: HModel.c,v 1.3 2015/10/12 12:07:24 cz277 Exp $
HCUDA      3.5.0      CUED   12/10/15  : $Id: HCUDA.cu,v 1.0 2015/10/12 12:07:23 cz277 Exp $
HANNet     3.5.0      CUED   12/10/15  : $Id: HANNet.c,v 1.0 2015/10/12 12:07:24 cz277 Exp $
HParm      3.5.0      CUED   12/10/15  : $Id: HParm.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HDict      3.5.0      CUED   12/10/15  : $Id: HDict.c,v 1.1.1.1 2006/10/11 09:54:57 jal58 Exp $
HNet       3.5.0      CUED   12/10/15  : $Id: HNet.c,v 1.2 2015/12/18 18:18:18 xl207 Exp $
HRec       3.5.0      CUED   12/10/15  : $Id: HRec.c,v 1.2 2015/10/12 12:07:24 cx277 Exp $
HUtil      3.5.0      CUED   12/10/15  : $Id: HUtil.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HAdapt     3.5.0      CUED   12/10/15  : $Id: HAdapt.c,v 1.3 2015/10/12 12:07:24 cz277 Exp $
HMap       3.5.0      CUED   12/10/15  : $Id: HMap.c,v 1.1.1.1 2006/10/11 09:54:57 jal58 Exp $
HNCache    3.5.0      CUED   12/10/15  : $Id: HNCache.c,v 1.0 2015/10/12 12:07:24 cz277 Exp $
```

# Compilation of HLMLib

1. Go to HLMLib:
```
$ cd ../HLMLib
```

2. Prepare compilation as follows:

Modify MakefileNVCC and replace any occurences of:

a) 'compute_75' to the version you might be using 'compute_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).

b) 'sm_75' to the version you might be using 'sm_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).
 
3. Compile HLMLib as follows:

a) First clean up:
```
HLMLib$ make -f MakefileNVCC clean
rm -f LModel.o LPMerge.o LPCalc.o LUtil.o LWMap.o LCMap.o LGBase.o HLMLib.a
```

b) Then compile:
```
HLMLib$ make -f MakefileNVCC all
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LModel.o LModel.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LPMerge.o LPMerge.c
In file included from LPMerge.c:48:
LPMerge.h:46: warning: "MAX_LMODEL" redefined
   46 | #define MAX_LMODEL    32
      |
In file included from LModel.h:41,
                 from LPMerge.c:46:
../HTKLib/HLM.h:92: note: this is the location of the previous definition
   92 | #define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */
      |
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LPCalc.o LPCalc.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LUtil.o LUtil.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LWMap.o LWMap.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LCMap.o LCMap.c
/usr/local/cuda/bin/nvcc -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -DSANITY -I. -I../HTKLib/   -c -o LGBase.o LGBase.c
if [ -f HLMLib.a ] ; then  /bin/rm HLMLib.a ; fi
ar rv HLMLib.a LModel.o LPMerge.o LPCalc.o LUtil.o LWMap.o LCMap.o LGBase.o
ar: creating HLMLib.a
a - LModel.o
a - LPMerge.o
a - LPCalc.o
a - LUtil.o
a - LWMap.o
a - LCMap.o
a - LGBase.o
ranlib HLMLib.a
```

c) The directory contents should look as follows:
```
HLMLib$ ls
HLMLib.a  LCMap.o   LGBase.o  LModel.o  LPCalc.o   LPMerge.o  LUtil.o  LWMap.o      MakefileNVCC
LCMap.c   LGBase.c  LModel.c  LPCalc.c  LPMerge.c  LUtil.c    LWMap.c  MakefileCPU
LCMap.h   LGBase.h  LModel.h  LPCalc.h  LPMerge.h  LUtil.h    LWMap.h  MakefileMKL
```

#. Compilation of HLMTools

1. Go to HLMTools:
```
HLMLib$ cd ../HLMTools
```

2. Prepare compilation as follows:

Modify MakefileNVCC and replace any occurences of:

a) 'compute_75' to the version you might be using 'compute_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).

b) 'sm_75' to the version you might be using 'sm_XX' - (currently for CUDA 11.2 and CUDNN 8.1.1 the version is 75).
 
3. Compile HLMTools as follows:

a) First clean up:
```
HLMTools$ make -f MakefileNVCC clean
rm -f *.o
```

b) Then compile:
```
HLMTools$ make -f MakefileNVCC all
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o Cluster -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  Cluster.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from Cluster.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 Cluster ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o HLMCopy -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  HLMCopy.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from HLMCopy.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
In file included from HLMCopy.c:55:
../HLMLib/LPMerge.h:46: warning: "MAX_LMODEL" redefined
   46 | #define MAX_LMODEL    32
      |
In file included from ../HLMLib/LModel.h:41,
                 from HLMCopy.c:53:
../HTKLib/HLM.h:92: note: this is the location of the previous definition
   92 | #define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */
      |
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 HLMCopy ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LAdapt -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LAdapt.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LAdapt.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
In file included from LAdapt.c:48:
../HLMLib/LPMerge.h:46: warning: "MAX_LMODEL" redefined
   46 | #define MAX_LMODEL    32
      |
In file included from ../HLMLib/LModel.h:41,
                 from LAdapt.c:46:
../HTKLib/HLM.h:92: note: this is the location of the previous definition
   92 | #define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */
      |
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LAdapt ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LBuild -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LBuild.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LBuild.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LBuild ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LFoF -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LFoF.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LFoF.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LFoF ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LGCopy -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LGCopy.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LGCopy.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LGCopy ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LGList -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LGList.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LGList.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LGList ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LGPrep -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LGPrep.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LGPrep.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LGPrep ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LLink -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LLink.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LLink.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LLink ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LMerge -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LMerge.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from LMerge.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
In file included from LMerge.c:55:
../HLMLib/LPMerge.h:46: warning: "MAX_LMODEL" redefined
   46 | #define MAX_LMODEL    32
      |
In file included from ../HLMLib/LModel.h:41,
                 from LMerge.c:53:
../HTKLib/HLM.h:92: note: this is the location of the previous definition
   92 | #define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */
      |
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LMerge ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LNewMap -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LNewMap.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LNewMap.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LNewMap ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LNorm -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LNorm.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from LNorm.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
In file included from LNorm.c:54:
../HLMLib/LPMerge.h:46: warning: "MAX_LMODEL" redefined
   46 | #define MAX_LMODEL    32
      |
In file included from ../HLMLib/LModel.h:41,
                 from LNorm.c:52:
../HTKLib/HLM.h:92: note: this is the location of the previous definition
   92 | #define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */
      |
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LNorm ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LPlex -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LPlex.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from LPlex.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
In file included from LPlex.c:55:
../HLMLib/LPMerge.h:46: warning: "MAX_LMODEL" redefined
   46 | #define MAX_LMODEL    32
      |
In file included from ../HLMLib/LModel.h:41,
                 from LPlex.c:53:
../HTKLib/HLM.h:92: note: this is the location of the previous definition
   92 | #define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */
      |
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LPlex ../bin.gpu ; fi
if [ ! -d ../bin.gpu -a X_ = X_yes ] ; then mkdir -p ../bin.gpu ; fi
/usr/local/cuda/bin/nvcc -o LSubset -m64 -ccbin gcc -gencode arch=compute_75,code=sm_75 -D'ARCH="x86_64"' -DCUDA -D_SVID_SOURCE -DOSS_AUDIO -I../HTKLib -I../HLMLib  LSubset.c ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -L/usr/X11R6/lib ../HTKLib/HTKLib.a ../HLMLib/HLMLib.a -lcudart -lcublas -lpthread -lm
In file included from /usr/include/x86_64-linux-gnu/bits/libc-header-start.h:33,
                 from /usr/include/stdio.h:27,
                 from ../HTKLib/HShell.h:43,
                 from LSubset.c:37:
/usr/include/features.h:187:3: warning: #warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE" [-Wcpp]
  187 | # warning "_BSD_SOURCE and _SVID_SOURCE are deprecated, use _DEFAULT_SOURCE"
      |   ^~~~~~~
if [ X_ = X_yes ] ; then /usr/bin/install -c -m 755 LSubset ../bin.gpu ; fi
```

c) The directory contents should look as follows:
```
HLMTools$ ls
Cluster    HLMCopy.c  LBuild    LFoF.c    LGList    LGPrep.c  LMerge    LNewMap.c  LPlex    LSubset.c    MakefileNVCC
Cluster.c  LAdapt     LBuild.c  LGCopy    LGList.c  LLink     LMerge.c  LNorm      LPlex.c  MakefileCPU
HLMCopy    LAdapt.c   LFoF      LGCopy.c  LGPrep    LLink.c   LNewMap   LNorm.c    LSubset  MakefileMKL
```

d) Test if HLMTools compiled by issuing:
```
HLMTools$ ./LBuild

USAGE: LBuild [options] wordMap langModel gramfile ....

 Option                                       Default

 -c n c  set cutoff for n-gram to c           1
 -d n c  set weighted discount pruning to c   off
 -f s    set output LM format to s            BIN
 -k n    set discounting range to [1..n]      7
 -l fn   build from existing LM from fn       off
 -n n    set model order                      max
 -t fn   load FoF table from fn               off
 -u n    set unigram floor count to n         1
 -x      save model with counts               off
 -A      Print command line arguments         off
 -C cf   Set config file to cf                default
 -D      Display configuration variables      off
 -S f    Set script file to f                 none
 -T N    Set trace flags to N                 0
 -V      Print version information            off 
```

A sample version output:
```
./LBuild -V

HTK Version Information
Module     Version    Who    Date      : CVS Info
LBuild     3.5.0      CUED   12/10/15  : $Id: LBuild.c,v 1.1.1.1 2006/10/11 09:54:44 jal58 Exp $
HShell     3.5.0      CUED   12/10/15  : $Id: HShell.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HMem       3.5.0      CUED   12/10/15  : $Id: HMem.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HMath      3.5.0      CUED   12/10/15  : $Id: HMath.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HWave      3.5.0      CUED   12/10/15  : $Id: HWave.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
HLabel     3.5.0      CUED   12/10/15  : $Id: HLabel.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $
LUtil      3.5.0      CUED   12/10/15  : $Id: LUtil.c,v 1.1.1.1 2006/10/11 09:54:43 jal58 Exp $
LWMap      3.5.0      CUED   12/10/15  : $Id: LWMap.c,v 1.1.1.1 2006/10/11 09:54:43 jal58 Exp $
LGBase     3.5.0      CUED   12/10/15  : $Id: LGBase.c,v 1.1.1.1 2006/10/11 09:54:43 jal58 Exp $
LModel     3.5.0      CUED   12/10/15  : $Id: LModel.c,v 1.1.1.1 2006/10/11 09:54:43 jal58 Exp $
LPCalc     3.5.0      CUED   12/10/15  : $Id: LPCalc.c,v 1.1.1.1 2006/10/11 09:54:43 jal58 Exp $
```

# Now we can get cracking - more to follow soon

# Thanks to
http://htk.eng.cam.ac.uk/

# Last Modified
2021/03/15 12:30 pm

# Contact

If you have any questions please do not hesitate to contact me on: silesiaresearch at gmail dot com



