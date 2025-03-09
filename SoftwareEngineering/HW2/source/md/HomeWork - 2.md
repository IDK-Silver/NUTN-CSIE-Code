# HomeWork - 2

## Kuwa軟體專案系統安裝及群聊應用

> 班級 ：資工三
>
> 姓名 ： 黃毓峰
>
> 學號 ： S11159005

## **一、**Kuwa軟體專案系統安裝

#### OS 資訊

- Host

   - Debian - 12

      - Linux 6.1.0-23-amd64

- Client (VM)

   - Ubuntu 22.04

### Step 1 - 設定VM

在 2025/03/08 日前 **[genai-os](https://github.com/kuwaai/genai-os) 在 Debain 是無法正常執的，不論是用 Docker Hub 所提供的 Image 或是自行根據 build.sh 來構建 Image 都是無發正常執行的**

為了能夠正常執行只好無奈妥協利用 VM 來架設 Repo 上所提供的開法環境，也就是要建立一個 OS 為 Ubuntu 22.04 的 VM

這裡會有一個問題就是要使VM能夠使用到 GPU ，所以要把 Host 的 GPU 給 passthrough 去 VM 裡面，這樣在使用 LLM 生成 Token 速度才不會太慢

這邊使用的 QEMU/KVM  來建立 VM，建立 VM 的過程不在這邊詳細演示網路上有許多的教學文章，這邊主要說明如何讓 GPU passthrough 去 VM 裡面

1. **啟用IOMMU**

   編輯 `/etc/default/grub.` 當中的  `GRUB_CMDLINE_LINUX_DEFAULT`  在原本的基礎上加上 `intel_iommu=on iommu=pt video=efifb:off` 最後應該會像是底下這樣

   ```cpp
   GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt video=efifb:off"
   ```

   

   更新 grub 

   ```cpp
   sudo grub-mkconfig -o /boot/grub/grub.cfg
   ```

   重新開機

   ```cpp
   sudo reboot
   ```



2. 在設定檔裡面新增 GPU 到 VM

   在這邊使用的是 [virt-manager](https://virt-manager.org) 如果是使用 Server 有桌面環境的話可以直接查詢 APP 名稱或是在 Terminal 當中打上

   ```cpp
   virt-manger
   ```

    如果沒有桌面環境可以考慮使用 x11 forward 來設定，或是使用如 [cockpit](https://cockpit-project.org) 等網頁介面來管裡

   底下是 cockpit 的 VM 設定介面

   ![cockpit_nv_device.png](./HomeWork%20-%202-assets/cockpit_nv_device.png)

3. 設定 VM 開啟及關閉時的 hooks

   為了能夠使 GPU 能夠給 VM 我們需要 GPU 從 host 端卸載掉, 可是這樣手動很麻煩，所以這邊設定 hook 當 VM 啟動時自動從 host 卸載掉 ，關閉後在自動加載

   1. 用**`sudo lsmod | grep nvidia`**確認目前有用到哪些核心模組

   2. 記錄下 pci

      ![cockpit_nv_device_pci_id.png](./HomeWork%20-%202-assets/cockpit_nv_device_pci_id.png)

   3. 建立 script 檔案

      ```cpp
      VM="你的 VM 名稱"
      
      sudo mkdir -p /etc/libvirt/hooks/qemu.d/${VM}/prepare/begin
      sudo touch /etc/libvirt/hooks/qemu.d/${VM}/prepare/begin/start.sh
      sudo chmod +x /etc/libvirt/hooks/qemu.d/${VM}/prepare/begin/start.sh
      sudo mkdir -p /etc/libvirt/hooks/qemu.d/${VM}/release/end
      sudo touch /etc/libvirt/hooks/qemu.d/${VM}/release/end/stop.sh
      sudo chmod +x /etc/libvirt/hooks/qemu.d/${VM}/release/end/stop.sh
      ```

   4. 編輯 start.sh

      ```cpp
      sudo nano /etc/libvirt/hooks/qemu.d/${VM}/prepare/begin/start.sh
      ```

      貼上底下 script

      ```cpp
      #!/bin/bash
      set -x
      
      # 停止顯示管理器
      systemctl stop sddm
      
      # Wayland下需要停止KDE Plasama服務
      #systemctl --user -M "你的使用者名稱" stop plasma-plasmashell.service
      
      # Unbind VTconsoles
      echo 0 > /sys/class/vtconsole/vtcon0/bind
      echo 0 > /sys/class/vtconsole/vtcon1/bind
      
      # Unbind EFI Framebuffer
      echo efi-framebuffer.0 > /sys/bus/platform/drivers/efi-framebuffer/unbind
      
      # 停止Nvidia服務
      systemctl stop nvidia-persistenced.service
      
      sleep 2
      
      # 取消載入NVIDIA核心模組
      modprobe -r nvidia_drm drm_kms_helper nvidia_modeset nvidia drm video
      
      sleep 2
      
      # 從宿主機移除GPU裝置和GPU音訊裝置
      virsh nodedev-detach pci_0000_01_00_0
      virsh nodedev-detach pci_0000_01_00_1
      
      # 載入VFIO核心模組
      modprobe vfio-pci
      
      ```

   5. 編輯 end.sh

      ```cpp
      sudo nano/etc/libvirt/hooks/qemu.d/${VM}/release/end/stop.sh
      ```

      貼上底下 script

      ```cpp
      #!/bin/bash
      set -x
      
      # 將GPU裝置加回宿主機
      virsh nodedev-reattach pci_0000_01_00_0
      virsh nodedev-reattach pci_0000_01_00_1
      
      # 取消載入VFIO核心模組
      modprobe -r vfio-pci
      
      # Rebind framebuffer to host
      echo "efi-framebuffer.0" > /sys/bus/platform/drivers/efi-framebuffer/bind
      
      # 載入NVIDIA核心模組
      modprobe nvidia_drm
      modprobe drm_kms_helper
      modprobe nvidia_modeset
      modprobe nvidia
      modprobe drm
      modprobe video
      
      
      # 啟動Nvidia服務
      systemctl start nvidia-persistenced.service
      
      # Bind VTconsoles
      echo 1 > /sys/class/vtconsole/vtcon0/bind
      echo 1 > /sys/class/vtconsole/vtcon1/bind
      
      
      ```

### Step 2 - 下載 **genai-os**

> 接下來的操作都是在VM裡面執行

#### 下載 Repo

1. 下載 Repo

   ```cpp
   git clone https://github.com/kuwaai/genai-os.git
   ```

2. 進入 docker 資料夾

   ```cpp
   cd genai-os/docker/
   ```

#### 設定 Docker 以及 Cuda

1. Install CUDA Driver

   ```cpp
   # Install the header of current running kernel
   sudo apt install -y linux-headers-$(uname -r)
   # or for auto-upgrade
   sudo apt install -y linux-headers-generic
   
   # Install the keyring
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
   wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   rm cuda-keyring_1.1-1_all.deb
   sudo apt update
   
   # Install the NVIDIA driver without any X Window packages
   sudo apt install -y --no-install-recommends cuda-drivers
   sudo reboot
   
   # Verify the version of installed driver
   cat /proc/driver/nvidia/version
   # Output sample:
   # NVRM version: NVIDIA UNIX x86_64 Kernel Module  545.23.08  Mon Nov  6 23:49:37 UTC 2023
   # GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)
   ```

2. Install Docker & Docker Compose

   ```cpp
   # Add official GPG key
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   
   # Setup repository
   echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   
   # Install necessary package
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   
   # Enable the service
   sudo systemctl --now enable docker
   
   # Enable unattended-update
   cat << EOT | sudo tee /etc/apt/apt.conf.d/51unattended-upgrades-docker
   Unattended-Upgrade::Origins-Pattern {
       "origin=Docker";
   };
   EOT
   ```

3. Install NVIDIA Container

   ```cpp
   # Add official GPG key
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   
   # Setup repository
   echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   
   # Install necessary package
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   
   # Enable the service
   sudo systemctl --now enable docker
   
   # Enable unattended-update
   cat << EOT | sudo tee /etc/apt/apt.conf.d/51unattended-upgrades-docker
   Unattended-Upgrade::Origins-Pattern {
       "origin=Docker";
   };
   EOT
   ```



#### 設定 Image

由於 在 Docker Hub 在面的 Image 預設是 CPU Base 所以我們需要手動下載 Cuda 並把它設為 lastest 

```cpp
docker pull kuwaai/model-executor:v0.3.4-cu121
docker tag kuwaai/model-executor:v0.3.4-cu121 kuwaai/model-executor:latest
```

#### 設定 LLM

如過像要使用 gguf model 的話需要手動加增一個 container，這邊以 Llama-3.1 TAIDE 來舉例

1. 下載模型

   ```bash
   wget https://huggingface.co/tetf/Llama-3.1-TAIDE-LX-8B-Chat-GGUF/resolve/main/Llama-3.1-TAIDE-LX-8B-Chat-F16.gguf
   ```

   取得模型位置

   ```bash
   realpath Llama-3.1-TAIDE-LX-8B-Chat-F16.gguf
   ```

2. 新增 docker-compose file

   ```bash
   nano compose/llama-3.1-taide-lx-8b-chat.yaml
   ```

   在輸入底下的配置文件，要注意的是 volumes 冒號左邊的是模型檔案位置

   ```cpp
   services:
     llama-3.1-taide-8b-chat-fp16-executor:
       image: kuwaai/model-executor
       environment:
         EXECUTOR_TYPE: llamacpp
         EXECUTOR_ACCESS_CODE: llama-3.1-taide
         EXECUTOR_NAME: Llama 3.1 TAIDE LX 8B Chat FP16
         EXECUTOR_IMAGE: TAIDE.png  # Refer to src/multi-chat/public/images
       depends_on:
         - executor-builder
         - kernel
         - multi-chat
       command: ["--model_path", "/var/model/Llama-3.1-TAIDE-LX-8B-Chat-F16.gguf", "--ngl", "-1", "--temperature", "0"]
       restart: unless-stopped
       volumes: ["/home/idk/models/Llama-3.1-TAIDE-LX-8B-Chat-F16.gguf:/var/model/Llama-3.1-TAIDE-LX-8B-Chat-F16.gguf"]
       deploy:
         resources:
           reservations:
             devices:
             - driver: nvidia
               device_ids: ['0']
               capabilities: [gpu]
       networks: ["backend"]
   ```

#### 設定 run.sh

在 run.sh 當中新增 `llama-3.1-taide-lx-8b-chat `應該會跟下面的內容一樣

```bash
#!/bin/bash

# Define the configuration array
confs=(
  "base"
  "dev" # Increase the verbosity for debug
  "pgsql"
  "copycat"
  "sysinfo"
  "pipe"
  "uploader"
  "token_counter"
  "gemini"
  "chatgpt"
  "dall-e"
  "llama3.1-70b-groq"
  "docqa"
  "searchqa"
  "llama-3.1-taide-lx-8b-chat"
)

# Append "-f" before each element
for i in "${confs[@]}"; do
  new_confs+=("-f" "compose/${i}.yaml" )
done

# Join the elements with white space
joined_confs=$(echo "${new_confs[@]}" | tr ' ' '\n' | paste -sd' ' -)

subcommand="${@:-up --remove-orphans}"
command="docker compose --env-file ./.env ${joined_confs} ${subcommand}"

echo "Command: ${command}"
bash -c "${command}"

```

#### 最後實際跑Container

> 這邊建議使用 tmux 來確保就算ssh 中斷也一樣會跑 Container

```bash
./run.sh
```

## **二、群聊應用**

- 可以透過 Create Room 並選擇 LLM 來對話

   ![CleanShot 2025-03-09 at 16.58.19@2x.png](./HomeWork%20-%202-assets/CleanShot%202025-03-09%20at%2016.58.19@2x.png)

- 可以選擇指定的LLM進行對話

   ![CleanShot 2025-03-09 at 16.59.32@2x.png](./HomeWork%20-%202-assets/CleanShot%202025-03-09%20at%2016.59.32@2x.png)

## **三、使用案例圖**

![CleanShot 2025-03-09 at 20.37.31@2x.png](./HomeWork%20-%202-assets/CleanShot%202025-03-09%20at%2020.37.31@2x.png)


