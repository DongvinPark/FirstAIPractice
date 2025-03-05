#   AI Practice
<br>

## Used Tensorflow, PyTorch, OpenCV
<br>

## Development Environment
- Used Asus Rog Strix Scar gaming laptop with NVIDIA Geforce RTX 3060 laptop GPU(TGP 140W)
- Ubuntu 22.04 LTS with 116 GB of total SSD space.
- Need to install nvidia gpu driver, cuda toolkit, cuDNN
- Need to install python, pytorch, tensorflow, opencv. Use pip cmd.
- Ask ChatGPT "how can I set image processing AI dev environment using pytorch, tensorflow, opencv?" and follow GPT's explanations.
- Used Jetbrains PyCharm Community edition IDE.
<br>

## How to Install Conda, Tensorflow, CUDA, cuDNN on Ubuntu 22.04 LTS with NVIDIA Geforce RTX GPU?
0. Things to Remember Before Getting Start
   1. NVIDIA GPU-related AI Libraries are very sensitive to version comparability issue.
   2. Nvidia-driver, CUDA toolkit, CuDNN's version must be comparable to each other.
   3. If this versions don't match each other, tensorflow might failed to recognize physical GPU(== NVIDIA Geforce RTX) or build python code.
<br><br/>
1. Install Fundamental Libraries(or Dependencies)
    1. I used this commands.
    ```shell
    sudo apt install -y build-essential cmake git unzip curl wget
    sudo apt install -y libatlas-base-dev libprotobuf-dev protobuf-compiler
    sudo apt install -y python3-dev python3-pip python3-venv python3-numpy
    ```
<br><br/>
2. Install Conda
   1. I strongly recommend to install Conda and ues its virtual environment because Conda is very useful on handling many python module's version dependencies.
   ```shell
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   2. Press 'enter' or type 'yes'.
   3. Activate 'conda' command on terminal and update Conda'
   ```shell
   source ~/.bashrc
   conda update conda
   ```
<br><br/>
3. Install Nvidia Driver
   1. I used this command to install 'nvidia-driver-550'. After installation was finished, reboot the Ubuntu.
   ```shell
   sudo apt install nvidia-driver-550
   sudo reboot
   ```
   2. After reboot, run this command. This will give us the information about the psysical GPU. 'CUDA 12.4' means, current nvidia-driver supports CUDA toolkit version 12.4. So we should install this CUDA toolkit version.
   ```shell
   dongvin@ubuntu >> nvidia-smi
   (cmd result) ..... CUDA 12.4 ...
   ```
<br><br/>
4. Install CUDA Toolkit
   1. Go to this NVIDIA official download url. We can also find this url by googling with 'cuda toolkit 12.4 download'.
   ```text
   https://developer.nvidia.com/cuda-12-4-0-download-archive
   ```
   2. Choose these categories : Linux >> x86_64 >> Ubuntu >> 22.04 >> runfile(local)
   3. NVIDIA official website will give you two commands. Run these commands.
   ```shell
   wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
   sudo sh cuda_12.3.0_545.23.06_linux.run
   ```
   4. About 30~40 seconds after running 2nd command, cuda toolkit installer will ask you to choose 'Abort' or 'Continue'. Choose 'Continue'.
   5. After that, list of items to be installed will appear. We must unselect 'nvidia-driver' because we already installed it on previous stage. Then finally, choose 'Install'.
   6. Update the library path and reboot the Ubuntu.
   ```shell
   echo $PATH | grep cuda
   echo $LD_LIBRARY_PATH | grep cuda
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
<br><br/>
5. Install DUDA Deep Neural Network(==cuDNN)
   1. We must choose right version of cuDNN. We will use Tensorflow-2.16.1 and previously installed CUDA 12.3.
   2. So, We must choose CuDNN 8.9.x. We can get the [version comparability information](https://www.tensorflow.org/install/source#gpu) at this official tensorflow website.
   3. Access [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) offered by NVIDIA and choose the 'Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x' tab.
   4. Click this. >> Local Installer for Linux x86_64 (Tar). And download will start. I strongly recommend to download it as 'tar'. I tried 'deb' several times, but all tries eventually failed.
   5. Go to the directory where the downloaded file exists. And run these commands in order.
   ```shell
   tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
   sudo cp -P include/cudnn*.h /usr/local/cuda/include/
   sudo cp -P lib/libcudnn* /usr/local/cuda/lib64/
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```
   6. Verify cuDNN Installation. Run this command.
   ```shell
   cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```
   7. If you see output like this, it means installation was successful. And reboot the computer again.
   ```shell
   #define CUDNN_MAJOR 8
   #define CUDNN_MINOR 9
   #define CUDNN_PATCHLEVEL 7
   ...
   ```
   8. Update library path.
   ```shell
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
   9. Run these commands to check for nvidia-driver and CUDA toolkit.
   ```shell
   nvcc --version
   nvidia-smi
   ```
   10. If you see results like this, installation is successful.
   ![Image](https://github.com/user-attachments/assets/33e3d5bd-8e78-4b45-85e6-2da2da19b080)
<br><br/>
6. Setup Conda Virtual Environment
   1. Make new conda virtual environment to test and install Tensorflow, PyTorch, OpenCV and activate env.
   ```shell
   #define CUDNN_MAJOR 8
   #define CUDNN_MINOR 9
   #define CUDNN_PATCHLEVEL 7
   ...
   conda create --name dongvin_test_env python=3.12
   conda activate donvin_test_env
   ```
<br><br/>
7. Install Tensorflow, PyTorch, OpenCV in conda env.
   1. We must install 'Tensorflow 2.16.1'. Use these commands.
   ```shell
   pip install tensorflow[and-cuda]==2.16.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install opencv-python opencv-python-headless
   ```
<br><br/>
8. Test Codes in This Project
   1. Copy the '06-handwritten-digits.py' and 07-AI-tool-version-check.py.
   2. And run each python file with this command.
   3. You can see the version information of Tensorflow, PyTorch, OpenCV, and physical GPU detected by Tensorflow.
   4. Example result of 'python3 07-AI-tool-version-check.py'
   ```text
    TensorFlow version: 2.16.1
   CUDA enabled: True
   OpenCV version: 4.11.0
   GPU detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   cuDNN version: 8
   ```