# ![Logo of diveXplore.](/assets/diveXplore.png) 
# DiveXplore Backend
This is the backend for diveXplore, our open-source video retrieval software. This section of the project handles all data preprocessing. The processed data is then utilized by the middleware and frontend of diveXplore to deliver a seamless user experience.

## 🚀 Getting Started 
To get the backend up and running, please follow the steps below.

### Prerequisites
Ensure you have the following software installed and running on your system:
- FFMPEG
- MongoDB

## Installation
**1. Clone the repository**

```bash
git clone https://github.com/marleo/divexplorebackend.git
cd divexplore-backend
```
**2. Install the required Python dependencies:**
   
```bash
pip install -r requirements.txt
```

## ⚙️ Usage
Once the Setup is complete, you can start analyzing your videos.

**1. Video Analysis**
   
This step initiates the video processing pipeline. The provided shell script will automatically execute all the necessary analysis scripts in the recommended order.
From the root directory of the project, execute the following commands:
```bash
chmod +x backend/analysis/process_video.sh
./backend/analysis/process_videos.sh [path_to_video]
```
**2. OpenCLIP Feature Extraction**

After the initial analysis, this script generates OpenCLIP features for each keyframe that was extracted.
```bash
python3 backend/extract_openclip_image_features.py
```
**3. Database Integration**
   
The final step is to collate all the generated information and add it to your MongoDB instance.
```bash
python3 backend/combine_analysis_files.py output/video_fps.txt output/scenes/ output/ocr/ output/asr/ output/summaries/
```
You have now successfully analyzed all your videos and populated the database!

--- 
🔗Frontend
Looking for the user-facing parts of diveXplore? You can find the frontend at the following repository: https://github.com/klschoef/divexplore
