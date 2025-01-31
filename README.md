📌 Overview
This project implements video cartoonization using Generative Adversarial Networks (GANs), specifically CycleGAN and CartoonGAN. The model transforms real-world video frames into cartoon-like animations, preserving motion consistency and artistic effects.

🚀 Features
✔ Converts real videos into high-quality cartoon animations.
✔ Uses CycleGAN/CartoonGAN for style transfer.
✔ Frame-wise processing for smooth motion consistency.
✔ Supports multiple cartoon styles.
✔ Implemented in Python, leveraging TensorFlow/PyTorch.

.

🛠 Technologies Used
Deep Learning: GANs (CycleGAN, CartoonGAN)
Frameworks: TensorFlow, PyTorch
Libraries: OpenCV, NumPy, Matplotlib
Video Processing: FFmpeg, OpenCV
Cloud & GPU: Google Colab, CUDA

Project Structure
├── dataset/                   # Training dataset (real & cartoon images)  
├── models/                    # Pre-trained GAN models  
├── scripts/                   # Training & inference scripts  
├── results/                   # Cartoonized video outputs  
├── requirements.txt           # Dependencies  
├── README.md                  # Project documentation  



#install dependencies

pip install -r requirements.txt

python cartoonize.py --input video.mp4 --output cartoonized.mp4 --model CycleGAN


python train.py --dataset dataset/ --epochs 100

🔬 Results
🚀 Before & After (Side-by-Side Comparisons)
🖼️ Sample outputs of cartoonized videos

📝 Future Work
✅ Enhance video smoothness using temporal consistency.
✅ Experiment with other GAN architectures.
✅ Improve model efficiency for real-time processing.

📩 Contribution
Contributions are welcome! Feel free to submit issues or pull requests.
