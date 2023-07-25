# YOLO_on_CUDA
You only look once - on CUDA!

Motivation:
Realtime object detection on the NVIDIA Jetson Nano

What I've tried:
0) OpenCV's dnn headers
   Problem: Jetson Nano does not have the OpenCV implemented CUDA backend
1) PyTorch
   Problem: Needs 6 GB of memory (Jetson Nano has 4 GB). And no, I'm not going to use virtual memory
2) TF Lite
   Problem: Same as that of PyTorch

Lucky for me though:

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html

AND

https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_custom_YOLO.html



   
