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

## How to Install Tensorflow, CUDA, cuDNN on Ubuntu 22.04 LTS with NVIDIA GPU?
0. Install git, python3, venv, numpy, cmake, gcc first
   1. use this cmd : 
   ```shell
    sudo apt install -y build-essential cmake git unzip curl wget
    sudo apt install -y libatlas-base-dev libprotobuf-dev protobuf-compiler
    sudo apt install -y python3-dev python3-pip python3-venv python3-numpy
    ```
1. Choose the tensorflow version first. this project uses tensorflow 2.18.0
   1. [official Tensorflow, CUDA, cuDNN version comparability info table](https://www.tensorflow.org/install/source#gpu)
2. Install CUDA 12.5
   1. run this cmd
   ```shell
    sudo apt install cuda-12-5
    ```
   2. if fails, ask ChatGPT about this issue with terminal console messages.
   3. this installation cmd will work.
   4. after installation was finished, reboot the computer.
   5. check the installation status by commands bellow.
   ```shell
    nvcc --version
    nvidia-smi
    ```
   6. these cmds will show result like this. If you see it, you successfully finished CUDA installation.
   ![Image](https://github.com/user-attachments/assets/33e3d5bd-8e78-4b45-85e6-2da2da19b080)
   7. reboot computer, and install tensorflow with physical GPU option.
   ```shell
    pip install tensorflow[and-cuda]
    ```
3.  Install cuDNN 9.3.0
    1. search the download url by googling with this word 'cuDNN 9.3.0 download'.
    2. click the NVIDIA download archive url.
    2. you may need to sign up NVIDIA developer and sign in.
    3. choose the right env(x86-64, ubuntu 22.04 ...) and follow the instructions provided by NVIDIA.
4. Install PyTorch and OpenCV
   1. run this cmd.
   ```shell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python opencv-python-headless
    ```
5. Install PyCharm and Setup Interpreter
   1. download PyCharm community version and open it.
   2. go to File >> Setting >> Project:FirstAIPractice >> Python Interpreter
   3. select interpreter Python3.10 in this directory : /usr/bin/python3.10
   4. run 07-AI-tool-version-check.py
   5. you will see text like this. text may differ form this according to your installations.
   6. if 'GPU detected:' has empty list like this '[]', it means that your tensorflow has failed to recognize the physical GPU.
   ```text
    TensorFlow version: 2.18.0
    CUDA enabled: True
    OpenCV version: 4.11.0
    GPU detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    cuDNN version: 9
    ```
6. Error Handling
   1. If you got a error message saying,
   ```text
   Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0.
   ```
   2. delete pre-installed tensorflow and reinstall it with version of '2.16.x'.
   3. this error occurs because TensorFlow 2.18.0 expects cuDNN 9.3+, but your system had cuDNN 9.1.0 installed.
   4. uninstall Tensorflow & Keras and reinstall it.
   ```shell
   pip uninstall tensorflow keras
   pip install tensorflow==2.16.1 keras
   ```