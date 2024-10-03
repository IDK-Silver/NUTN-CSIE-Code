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

