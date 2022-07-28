# python repro

```
tar xvzf ./citrinet.tgz
python ./final_repo.py
```

# build

Set `$ORT_INSTALL_PREFIX` to onnxruntime install directory.
Set `$EIGEN_HDR_DIR` to eigen install prefix, eg `/usr/local/include`

```
mkdir build
cd build
cmake -DORT_INSTALL_PREFIX=$ORT_INSTALL_PREFIX -DEIGEN_HDR_DIR=$EIGEN_HDR_DIR ../src && make -j

# run
./main $TEST_MODEL
```
