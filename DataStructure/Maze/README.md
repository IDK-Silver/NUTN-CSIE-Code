# 迷宮工具

迷宮工具自行設定迷宮大小，選擇起始及終點並點選來設定可走路徑

安下按鈕可以求出最了路徑



# 編譯說明

### 需求套件說明

* C++11 以上的編譯器
* CMake >= 3.26
* Qt 6



### 實際系統套件版本

* Linux Kernel 6.6.1

* KDE Plasma 5.27.9

* GCC version 13.2.1

* CMake version 3.27.8

* Qt 6.6.0-3

* GNU Make 4.4.1

  

## ArchLinux

* Download Source Code
```
git clone https://github.com/IDK-Silver/NUTN_Code.git
```
* Install Qt6
```
sudo pacman -S qt6
```

* Install CMake
```
sudo pacman -S cmake
```

* Make directory 
```
cd NUTN_Code/DataStructure/Maze/
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
./Maze
```



## Windows

<u>在Windows 底下編譯使用 [MSYS ](https://www.msys2.org/) 來配置環境，其他方法請自行研究,</u>

<u>或是考慮用WSL按照Linux的編譯說明來用</u>



* 打開 MSYS 的界面並下載package

```
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-cmake mingw-w64-x86_64-qt6 git
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
cd NUTN_Code/DataStructure/Maze/
mkdir build 
cd build/
```

* 用CMake 生成 Makefile 並編譯

```
cmake ..
cmake --build .
```

* 執行軟體

```
Maze.exe
```

