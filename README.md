# build

Set `$ORT_INSTALL_PREFIX` to onnxruntime install directory.

```
mkdir build
cd build
cmake -DORT_INSTALL_PREFIX=$ORT_INSTALL_PREFIX ../src && make -j

# run
./main $TEST_MODEL
```
