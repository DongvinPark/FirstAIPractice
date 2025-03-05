# AI Practice

## Used TensorFlow, PyTorch, OpenCV

## Development Environment
- Using an **Asus ROG Strix Scar** gaming laptop with an **NVIDIA GeForce RTX 3060 Laptop GPU (TGP 140W)**.
- **Ubuntu 22.04 LTS** with **116 GB of total SSD space**.
- Need to install the **NVIDIA GPU driver, CUDA Toolkit, and cuDNN**.
- Need to install **Python, PyTorch, TensorFlow, and OpenCV** using `pip`.
- Ask ChatGPT:  
  _"How can I set up an image processing AI development environment using PyTorch, TensorFlow, and OpenCV?"_  
  and follow its explanations.

---

## How to Install Conda, TensorFlow, CUDA, and cuDNN on Ubuntu 22.04 LTS with an NVIDIA GeForce RTX GPU  

### 0. Things to Remember Before Getting Started  
1. **NVIDIA GPU-related AI libraries are highly sensitive to version compatibility issues**.  
2. **NVIDIA driver, CUDA Toolkit, and cuDNN versions must be compatible** with each other.  
3. If these versions don’t match, **TensorFlow might fail to recognize the physical GPU (NVIDIA GeForce RTX) or fail to build Python code**.  

---

### 1. Install Fundamental Libraries (Dependencies)  
Run the following commands:  
```shell
sudo apt install -y build-essential cmake git unzip curl wget
sudo apt install -y libatlas-base-dev libprotobuf-dev protobuf-compiler
sudo apt install -y python3-dev python3-pip python3-venv python3-numpy
```

---

### 2. Install Conda  
1. **I strongly recommend installing Conda** and using its virtual environment, as Conda is very useful for managing Python module dependencies.  
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh
```  
2. Press **Enter** or type **yes** when prompted.  
3. Activate the `conda` command in the terminal and update Conda:  
```shell
source ~/.bashrc
conda update conda
```

---

### 3. Install NVIDIA Driver  
1. Run the following command to install **nvidia-driver-550**. After installation, reboot Ubuntu.  
```shell
sudo apt install nvidia-driver-550
sudo reboot
```  
2. After rebooting, verify the installation by running:  
```shell
nvidia-smi
```  
If the output includes `CUDA 12.3`, that means the installed NVIDIA driver supports CUDA Toolkit **12.3**, so we should install this specific version.

---

### 4. Install CUDA Toolkit  
1. Go to the **official NVIDIA CUDA download page**:  
   👉 [CUDA Toolkit 12.3 Download](https://developer.nvidia.com/cuda-12-3-0-download-archive)  
   _(Alternatively, search "CUDA Toolkit 12.3 download" on Google.)_  
2. Select the following options:  
   - **Linux** → **x86_64** → **Ubuntu** → **22.04** → **runfile (local)**  
3. Run the following commands provided by NVIDIA:  
```shell
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run
```  
4. About **30–40 seconds after running the second command**, the installer will ask you to choose **Abort** or **Continue**. Select **Continue**.  
5. A list of installation options will appear. **Unselect "NVIDIA Driver"**, as we have already installed it in the previous step. Then, proceed with the installation.  
6. Update the library path and reboot Ubuntu:  
```shell
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### 5. Install CUDA Deep Neural Network (cuDNN)  
1. We must choose the **correct cuDNN version** that matches TensorFlow **2.16.1** and CUDA **12.3**.  
2. **Check version compatibility** 👉 [TensorFlow GPU Support Table](https://www.tensorflow.org/install/source#gpu)  
3. Visit the **cuDNN Archive** 👉 [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)  
4. Select **"Download cuDNN v8.9.7 (December 5th, 2023) for CUDA 12.x"**.  
5. Download the **"Local Installer for Linux x86_64 (Tar)"** version (**DO NOT use the `.deb` package**; it often fails).  
6. Extract and install cuDNN:  
```shell
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
sudo cp -P include/cudnn*.h /usr/local/cuda/include/
sudo cp -P lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```  
7. Verify the cuDNN installation:  
```shell
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```  
8. Update the library path and reboot the system:  
```shell
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
9. Verify the NVIDIA-Driver, CUDA toolkit and cuDNN:
   1. If you get result like this, installation was successful.
   ![Image](https://github.com/user-attachments/assets/33e3d5bd-8e78-4b45-85e6-2da2da19b080)
```shell
nvcc --version
nvidia-smi
```

---

### 6. Set Up a Conda Virtual Environment  
1. Create a **new Conda virtual environment** and activate it:  
```shell
conda create --name dongvin_test_env python=3.12
conda activate dongvin_test_env
```

---

### 7. Install TensorFlow, PyTorch, and OpenCV in the Conda Environment  
1. Install **TensorFlow 2.16.1**, PyTorch, and OpenCV:  
```shell
pip install tensorflow[and-cuda]==2.16.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python opencv-python-headless
```

---

### 8. Test AI Libraries in This Project
1. Copy the test Python Files:
   - **06-handwritten-digits.py**
   - **07-AI-tool-version-check.py**
2. Run each Python file:
```shell
python3 06-handwritten-digits.py
python3 07-AI-tool-version-check.py
```
3. If everything is correctly installed, you should see output containing these information.
```text
TensorFlow version: 2.16.1
CUDA enabled: True
OpenCV version: 4.11.0
GPU detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
cuDNN version: 8
```

---

### ✅ Final Notes  
- If **TensorFlow does not detect the GPU**, check:  
  ```shell
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  ```  
- If **PyTorch does not detect the GPU**, check:  
  ```shell
  python -c "import torch; print(torch.cuda.is_available())"
  ```  
- If **any library fails**, verify versions using:  
  ```shell
  nvcc --version
  nvidia-smi
  
