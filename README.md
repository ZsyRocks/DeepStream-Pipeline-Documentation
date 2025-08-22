# DeepStream Pipeline Documentation

NVIDIA DeepStream is a high-performance streaming analytics toolkit designed for building AI-powered video analytics applications.

This project sets up a simple people detection and tracking pipeline using NVIDIA DeepStream 7.1. It simulates four parallel camera feeds by duplicating a single video input across four sources. This is ideal for testing and reproducibility.<br>

---

## 🧩 Features

- ✅ Custom People detection using a custom model that takes 4 batches
- ✅ Object tracking using multiple trackers (IOU, NvSORT, NvDeepSORT and NvDCF)
- ✅ Tiled display (2x2) showing 4 video feeds
- ✅ Output sink to save the processed video (MP4)
- ✅ Runs on DeepStream 7.1 + Ubuntu 22.04 on VM
<br>

---

## 🎁 Prerequisites


### To install cuDNN for DeepStream [OPTIONAL]

Install cuDNN 9.12 tar file using wget

```bash
sudo wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz
```

Extract the archive

```bash
tar -xvf cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz
```

Navigate to the cuDNN archive folder

```bash
cd cudnn-linux-x86_64-9.12.0.46_cuda12-archive
```

Copy cuDNN headers and libraries to CUDA include directory\

```bash
sudo cp include/cudnn*.h /usr/local/cuda/include/
sudo cp lib/libcudnn* /usr/local/cuda/lib64/
```

Give read permissions to cuDNN

```bash
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Create essential symbolic links

```bash
cd /usr/local/cuda/lib64
sudo ln -sf libcudnn.so.9.12.0 libcudnn.so.9
sudo ln -sf libcudnn.so.9 libcudnn.so
sudo ln -sf libcudnn_cnn.so.9.12.0 libcudnn_cnn.so.9
sudo ln -sf libcudnn_cnn.so.9 libcudnn_cnn.so
sudo ln -sf libcudnn_adv.so.9.12.0 libcudnn_adv.so.9
sudo ln -sf libcudnn_adv.so.9 libcudnn_adv.so
```

And finally update the linker cache

```bash
sudo ldconfig
```

---

## 🛠️ Version Dependencies 

![Ubuntu Version](Images/ubuntu_version.png)

![Versions](Images/versions.png) 
<br>
⬆️⬆️⬆️
With cuDNN 9.12 installed

![Nvidia Driver Version](Images/nvidia_driver.png) 

---


## 📥 Step-by-Step Instructions


### 1. Prepare the working environment/DIR

```bash
sudo apt update
```

Install gedit which is a notepad like environment for Linux Ubuntu

```bash
sudo apt install gedit -y
```

Change to DeepStream working DIR 

```bash
cd /opt/nvidia/deepstream/deepstream-7.1 
```
<br>

### 2. Prepare sample video and reID model(for some tracking configs)

Make sure to replace 'YourWindowsUser' and 'YourLinuxUser' accordingly

(assuming the sample video is under the 'Videos' Folder on your windows machine)
```bash
cp /home/YourLinuxUser/Downloads/myvideo.mp4 /home/YourLinuxUser/myvideo.mp4 
```

Create a new DIR to place the reID model to be used for tracking later on

```bash
sudo mkdir /opt/nvidia/deepstream/deepstream-7.1/samples/models/Tracker/
```

Downloading reID model .etlt file and saving it into the newly created DIR

```bash
sudo wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt' -P /opt/nvidia/deepstream/deepstream-7.1/samples/models/Tracker/
```
<br>

### 3. Create and edit the pipeline.txt 

```bash
sudo gedit pipeline.txt 
```

Instructions:
- Copy the contents of the 'pipeline.txt' and paste it inside
- Make sure to change the 'URI file' in each SOURCE section accordingly to where you place your video 
- Also change the 'output-file' in the SINK1 section accordingly to where you want to save your video
<br>

### 4. Prepare Custom detector and tracker

Enable permission to copy onnx/engine file into this DIR first

```bash
sudo chmod -R a+rw /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector
```

Make sure to replace 'YourLinuxUser' accordingly 

(assuming the custom detector is under the 'Downloads' Folder on your windows machine)

```bash
cp /home/YourLinuxUser/Downloads/best_b4.onnx /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/best_b4.onnx
```

Manually build the engine file using trtexec

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/best_b4.onnx \
  --saveEngine=/opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/best_b4.onnx_b4_gpu0_fp16.engine \
  --fp16
```

List available detectors

```bash
ls -lh /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/
```

![Available Detectors](Images/with_engine.png)

You should be able to see the custom detector with the '.engine' file built.
<br>

List available trackers

```bash
ls -lh /opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app
```
![Available Tracker Configs](Images/available_trackers.png)

You should be able to see only 6 different config_tracker yml files
<br>

### 5. Prepare the config_infer_primary.txt and parser

```bash
sudo gedit config_infer_primary.txt 
```

Instructions:
- Copy the contents of the 'config_infer_primary.txt' and paste it inside 

```bash
sudo gedit /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser/parser.cpp
```

Instructions:
- Copy the contents of the 'parser.cpp' and paste it inside 
<br>

Then recompile the parser into a .so file

```bash
sudo g++ -shared -fPIC -o /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser/libcustomparser.so "/opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser/parser.cpp"   `pkg-config --cflags --libs gstreamer-1.0`   -I/opt/nvidia/deepstream/deepstream-7.1/include   -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes   -I/usr/local/cuda/include
```
<br>

### 7. Run the pipeline and save the video

Run the pipeline

```bash
deepstream-app -c /opt/nvidia/deepstream/deepstream-7.1/pipeline.txt
```

You should be able to see the Deepstream interface like this 
<br>

![Interface](Images/deepstream_interface.png)
<br>

LEFT CLICK to zoom in to one video/source and RIGHT CLICK to zoom out
<br>

[OPTIONAL]
🔻🔻🔻

```bash
cp /home/YourLinuxUser/output_tiled.mp4 /mnt/c/Users/YourWindowsUser/Videos/output.mp4
```

To save the outtputed video onto your windows machine if on WSL (Make sure to change 'YourLinuxUser' and 'YourWindowsUser' accordingly)
<br>

### 8. Trying out multiple tracker configs (IOU, NvSORT, NvDeepSORT and NvDCF) provided by DeepStream

Change reID model paths in NvDeepSORT.yml and NvDCF_accuracy.yml

```bash
sudo gedit /opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml
```

![reID](Images/reID_paths.png)

#### Old path
/opt/nvidia/deepstream/deepstream/samples/models/Tracker/resnet50_market1501.etlt

#### Updated path 
/opt/nvidia/deepstream/deepstream7.1/samples/models/Tracker/resnet50_market1501.etlt
<br>
<br>
Do the same for NvDCF_accuracy.yml

```bash
sudo gedit /opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml
```

Check that the reID model exist

```bash
ls -lh /opt/nvidia/deepstream/deepstream-7.1/samples/models/Tracker/
```

![reID model](Images/reID_model.png) <br>

Now open the pipeline.txt

```bash
sudo gedit pipeline.txt
```

Scroll down to the TRACKER section of the pipeline.txt until you see the highlighted section below which we can observe 6 different tracker configs

![Tracker Configs](Images/tracker_configs.png)

Instruction:
- Only UNCOMMENT whichever tracker "ll-config-file" you want to use
- Leave only one "ll-config-file" UNCOMMENTED at a time and the rest Commented OUT
- Then run the pipeline
<br>

![reID model](Images/reID_model.png)

- The NvDCF_accuracy and NvDeepSORT tracker configs uses the reID model we installed earlier
- And as seen in the image we do not have an engine file built. 
- If you want to build the engine we need to install TAO-toolkit to build the engine from the reID model(.etlt file) otherwise it will run with default values or both NvDCF_accuracy and NvDeepSORT
<br>

### 9. Addtional Performance Optimisations

#### Open pipeline.txt, 

```bash
sudo gedit pipeline.txt
```

In the STREAMUX section 

- Set the height and width of streammux to the input resolution.

![STREAMMUX](Images/streammux.png)

In this case, the input resolution of my video is 1270x720
<br>

If you are streaming from live sources such as RTSP or from USB camera, set live-source=1 in [streammux] group of config file. This enables proper timestamping for live sources creating smoother playback

<br>
In the OSD and TILED_DSISPLAY sections 

- Disable them by setting 'enable=0'

![OSD](Images/osd.png)
&
![TILED](Images/tiled.png)

<br>
Finally, in the SINK0 and SINK1 sections

- Set 'type=1' to make them fakesinks

![SINKS](Images/sinks.png)

- Disabling OSD means there will be no bounding boxes drawn,
- Choosing fakesinks will get rid of the deeepstream interface and get rid of the proccessed Frames
- As tiling and visual output can take up GPU resource, we can disable them when rendering is not required and we want to run inference on the edge

---


## 📚 Resources and notes

- [NVIDIA DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)

### Notes:
- My Deepstream was installed from the tar package
- __The ouput videos using each tracker can be found inside the Videos folder in this repository__

### Tracker Performance Comparison

The table below summarizes the FPS performance of different trackers tested on the same video source in both the **Virtual Machine** and **WSL**.

#### VM

| **Trackers**         | **FPS Range across all sources**   | **Average FPS** |
|-----------------|----------------|------------------|
| IOU Tracker      | 24.59 – 21.90   | 23.32            |
| NvDCF_perf       | 24.46 – 21.62   | 23.05          |
| NvDCF_max_perf   | 24.87 – 21.39   | 22.67           |
| NvDCF_accuracy(used default values)   | 23.46 – 21.06   | 22.15           |
| NvSORT           | 24.30 – 20.81   | 21.96          |
| NvDeepSORT(used default values)      | 24.71 – 21.82   | 23.21            |

#### WSL

| **Trackers**         | **FPS Range across all sources**   | **Average FPS** |
|-----------------|----------------|------------------|
| IOU Tracker      | 37.4 – 36.27   | 36.86            |
| NvDCF_perf       | 36.94 - 35.90  | 36.20          |
| NvDCF_max_perf   | 37.38 – 36.37   | 36.59          |
| NvDCF_accuracy(used default values)   | 37.30 – 36.55   | 37.04           |
| NvSORT           | 37.32 – 36.61  | 36.86          |
| NvDeepSORT(used default values)      | 37.52 – 36.40   | 36.76           |

---
