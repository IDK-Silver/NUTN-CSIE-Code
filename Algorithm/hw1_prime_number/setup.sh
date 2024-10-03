#! /bin/bash
if ! command -v git &> /dev/null
then
    echo "git could not be found, please install git."
    exit
fi

# 檢查目錄是否存在
if [ -d "lib/big_int" ]; then
    echo "Directory 'lib/big_int' already exists. Skipping clone."
else
    git clone https://github.com/IDK-Silver/big-int.git lib/big_int
fi

cd lib/big_int
git pull
cd ../../

if ! command -v cmake &> /dev/null
then
    echo "cmake could not be found, please install cmake."
    exit
fi

if ! command -v make &> /dev/null
then
    echo "make could not be found, please install make."
    exit
fi

if ! command -v g++ &> /dev/null
then
    echo "gcc could not be found, please install gcc."
    exit
fi

# 檢查 build 目錄是否存在
if [ -d "build" ]; then
    echo "build 目錄已存在，正在清理..."
    rm -rf build
fi

# 創建 build 目錄
echo "正在創建 build 目錄..."
mkdir build
cd build

# 使用 CMake 生成構建文件
echo "正在使用 CMake 生成構建文件..."
cmake ..

# 使用 make 編譯
echo "正在使用 make 編譯..."
cmake --build . -j 8

# 返回上一級目錄
cd ..

echo "編譯完成。"


# 檢查是否安裝了 Python
if ! command -v python3 &> /dev/null
then
    echo "Python 未找到，請安裝 Python。"
    exit
fi

# 檢查是否安裝了 venv 模組
if ! python3 -c "import venv" &> /dev/null
then
    echo "venv 模組未找到，請安裝 venv。"
    exit
fi
# 檢查虛擬環境是否已存在
if [ -d "venv" ]; then
    echo "Python 虛擬環境已存在，跳過建立步驟。"
else
    echo "正在建立 Python 虛擬環境..."
    python3 -m venv venv
fi

# 啟動虛擬環境
source venv/bin/activate

# 安裝所需的 Python 套件
echo "正在安裝 pandas、matplotlib 和 seaborn..."
pip install pandas matplotlib seaborn

# 退出虛擬環境
deactivate

echo "Python 套件安裝完成。"

# 下載字體檔案
echo "正在下載字體檔案..."
# 檢查是否安裝了 wget
if ! command -v wget &> /dev/null
then

    echo "請手動安裝 wget 後再運行此腳本。"
    exit 1

fi

echo "確認 wget 已安裝。"

wget -O TaipeiSansTCBeta-Regular.ttf "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"

if [ $? -eq 0 ]; then
    echo "字體檔案下載成功。"
else
    echo "字體檔案下載失敗。"
fi

