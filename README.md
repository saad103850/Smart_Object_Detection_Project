# Smart Object Detection System

A Python and Google Colab based object detection and counting web application built with **Ultralytics YOLO11 (`yolo11n.pt`)** and **Gradio**. Users can upload an image, view bounding boxes with confidence scores, get a summary count of detected objects, download the annotated output, and launch an interactive web UI.

## Features

- **Fast Object Detection:** Uses the lightweight `yolo11n.pt` model for quick inference (supports GPU acceleration when available).
- **Visual Annotations:** Plots bounding boxes and labels directly onto detected objects.
- **Object Counting & Confidence Scores:** Summarizes counts per class, total detected objects, and individual confidence percentages.
- **Export Results:** Saves and downloads the annotated image (`object_detection_result.jpg`).
- **Interactive Gradio Interface:** Provides a browser-based UI with a shareable public link for testing images easily.

## Tech Stack

- **Python**
- **Ultralytics (`YOLO`)** — Object detection model (`yolo11n.pt`)
- **Gradio** — Interactive web interface
- **PyTorch, OpenCV, NumPy, Pillow, Matplotlib** — Tensor/GPU checks, image conversion, plotting, and visualization
- **Google Colab** — Notebook environment, file upload/download utilities

## How It Works

1. **Setup & Model Loading:** Installs `ultralytics` and `gradio`, checks GPU availability (`torch.cuda.is_available()`), and loads `yolo11n.pt`.
2. **Image Upload & Inference:** Uploads an image in Colab and runs `model.predict(..., conf=0.25)`.
3. **Visualization & Summary:** Renders bounding boxes (`result.plot()`), converts BGR to RGB for display, counts detected classes using `collections.Counter`, and prints confidence scores.
4. **Download & Web App:** Saves `object_detection_result.jpg` and launches a Gradio `Interface` returning both the annotated image and object count summary.

## Sample Output

Example detection output from a test run:

- **Detected Summary:** `person: 4`, `suitcase: 2`, `car: 2`, `backpack: 1` (**Total Objects:** `9`)
- **Inference Speed:** ~23.4 ms at `480x640` resolution (on Colab GPU)

## Usage (Google Colab)

1. Open the notebook in **Google Colab** and enable a **GPU** runtime (`Runtime` > `Change runtime type` > `GPU`).
2. Run the installation and import cell:
   ```bash
   !pip install -q ultralytics gradio
