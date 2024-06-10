# 簡介
根據課本的Absolute Loader的設計構想，並利用C來實做出來，該設計出的程式可以讀取以由 ACSII Code 編碼組成的一串16進位數值，並根據給予的記憶體起始位置輸出載入後的記憶體位置，與其改位置的記記憶體數值，可以選在在Shell or Console中顯示或是把結果寫入在文件。
 
 # 如何編譯
## Linux
*  GCC 可編譯 C11
* CMake 大於 3.0
* Make

```
git clone https://github.com/IDK-Silver/NUTN-CSIE-Code.git
```

```
 cd NUTN-CSIE-Code/SystemSoftware/absolute_loader/
```

```
mkdir build && cd build
```

```
cmake .. && cmake --build .
```
## Windows
用 MSYS2 或 MSVC

## MacOS
等我有MacBook再補
# 測試程式
把　../test_file/source.txt 用　Absoult Loader　做載入並把結果存到　 ../test_file/source.mem
```
./absolute_loader -A04096 ../test_file/source.txt -f
```