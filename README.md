# DeepStream Pipeline Documentation

NVIDIA DeepStream is a high-performance streaming analytics toolkit designed for building AI-powered video analytics applications.

This project sets up a simple people detection and tracking pipeline using NVIDIA DeepStream 7.1. It simulates four parallel camera feeds by duplicating a single video input across four sources. This is ideal for testing and reproducibility.<br>

---

## 🧩 Features

- ✅ Custom People detection using a custom model that takes 4 batches
- ✅ Object tracking using multiple trackers (IOU, NvSORT, NvDeepSORT and NvDCF)
- ✅ Tiled display (2x2) showing 4 video feeds
- ✅ Video output saved to file (MP4)
- ✅ Runs on DeepStream 7.1 + WSL2 (Windows Subsystem for Linux)
<br>

---

## 🛠️ Version Used 

![Ubuntu Version](Images/ubuntu_version.png)

![Versions](Images/versions.png)
<br>

---

## 📥 Step-by-Step Instructions


### 1. Prepare the working environment/DIR

```bash
sudo apt update && sudo apt upgrade
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
cp /mnt/c/Users/YourWindowsUser/Videos/myvideo.mp4 /home/YourLinuxUser/myvideo.mp4 
```

Create a new DIR to place the reID model to be used for tracking later on

```bash
sudo mkDIR /opt/nvidia/deepstream/deepstream-7.1/samples/models/Tracker/
```

Downloading reID model .etlt file and saving it into the newly created DIR

```bash
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt' -P /opt/nvidia/deepstream/deepstream-7.1/samples/models/Tracker/
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

Make sure to replace 'YourWindowsUser' and 'YourLinuxUser' accordingly 

(assuming the custom detector is under the 'Downloads' Folder on your windows machine)
```bash
cp /mnt/c/Users/YourWindowsUser/Downloads/best_b4.onnx /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/best_b4.onnx
```

List available detectors

```bash
ls -lh /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/
```

![Available Detectors](Images/without_engine.png)

You should be able to see the custom detector but there is no '.engine' file here, hence we need to build it.
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
### 6. Run the pipeline and build the engine file

Enable permission to write the built engine file into this DIR first

```bash
sudo chmod -R a+rw /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector
```
<br>
Then, run the pipeline

```bash
deepstream-app -c /opt/nvidia/deepstream/deepstream-7.1/pipeline.txt
```

You should be able to see the Deepstream interface like this 
<br>

![Interface](Images/deepstream_interface.png)

Let the engine build and after the pipeline run finish, check if the engine file is built

```bash
ls -lh /opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/
```

![Available Detectors](Images/with_engine.png)
<br>

As seen here, there is an engine file now, so whenever we run the pipeline, we don't have to rebuild the engine from scratch.

<br>
### 7. Edit the config_infer_primary again

```bash
sudo gedit config_infer_primary.txt
```

Instruction:
- Comment OUT the 'infer_dims' in the PROPERTY section
- Comment OUT the 'onnx_file' in the PROPERTY section
- UNCOMMENT the 'model-engine-file' in the PROPERTY section

<br>
### 8. Run the pipeline and save the video onto the windows machine

```bash
deepstream-app -c /opt/nvidia/deepstream/deepstream-7.1/pipeline.txt
```

The process should load much faster since we don't have to built the engine file
<br>

LEFT CLICK to zoom in to one video/source and RIGHT CLICK to zoom out
<br>

[OPTIONAL]<br>
🔻🔻🔻

```bash
cp /home/YourLinuxUser/output_tiled.mp4 /mnt/c/Users/YourWindowsUser/Videos/output.mp4
```

To save the outtputed video onto your windows machine. 
(Make sure to change 'YourLinuxUser' and 'YourWindowsUser' accordingly) 

<br>
### 9. Trying out multiple tracker configs (IOU, NvSORT, NvDeepSORT and NvDCF) provided by DeepStream

```bash
sudo gedit pipeline.txt
```

Scroll down to the TRACKER section of the pipeline.txt until you see the highlighted section below which we can observe SIX different tracker configs

![Tracker Configs](Images/tracker_configs.png)

Instruction:
- Only UNCOMMENT whichever tracker ll-config-file you want to use
- Leave only ONE ll-config-file UNCOMMENTED and the rest Commented OUT
- Then run the pipeline and save the video to your local
<br>

The NvDCF_accuracy and NvDeepSORT tracker configs uses the reID model we installed earlier but we do not have an engine file built. If you want to build the engine we need to install TAO-toolkit to build it from the reID model(.etlt file)

---


## 📚 Resources and notes

- [NVIDIA DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)

Notes:
My Deepstream was installed from the tar package

---
