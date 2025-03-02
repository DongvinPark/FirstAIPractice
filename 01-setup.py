import torch
import cv2

def main():
    print("PyTorch Version : ", torch.__version__)
    print("CUDA Available : ", torch.cuda.is_available())
    print("OpenCV Version : ", cv2.__version__)

if __name__ == "__main__":
    main()