# 計算機

可以用的機算機

支援以下運算

* `+`   加法
* `-`   減法
* `*`   乘法
* `/`   除法
* `()` 括號優先運算
* `^`   次方
* `%`   取餘數

數值上限 : `1.7E +/- 308 (15 位數)` 



# 編譯說明

### 需求套件說明

* C11 以上的編譯器

* CMake >= 3.26

  

### 實際系統套件版本

* Linux Kernel 6.6.1

* KDE Plasma 5.27.9

* GCC version 13.2.1

* CMake version 3.27.8

* GNU Make 4.4.1

  

## ArchLinux

* Download Source Code
```
git clone https://github.com/IDK-Silver/NUTN_Code.git
```
* Install CMake
```
sudo pacman -S cmake
```

* Make directory 
```
cd NUTN_Code/DataStructure/Calculator/
mkdir build 
cd build/
```

* Generate makefile and building
```
camke ..
cmake --build .
```

* Run APP
```
./Calculator
```



## Windows

<u>在Windows 底下編譯使用 [MSYS ](https://www.msys2.org/) 來配置環境，其他方法請自行研究,</u>

<u>或是考慮用WSL按照Linux的編譯說明來用</u>



* 打開 MSYS 的界面並下載package

```
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-cmake git
```

* 把 MSYS 的路徑加入 System Path
    ```
    C:\msys64\mingw64\bin
    ```
    ```
    C:\msys64\usr\bin
    ```

* 創件資料夾

```
cd NUTN_Code/DataStructure/Calculator/
mkdir build 
cd build/
```

* 用CMake 生成 Makefile 並編譯

```
camke ..
cmake --build .
```

* 執行軟體

```
Calculator.exe
```

