from flask import Flask, request, render_template, send_file, jsonify
import os
import mimetypes
from PIL import Image
import tempfile
import subprocess
from utils.transforms import get_no_aug_transform
import torch
from models.generator import Generator
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
import re

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and map to CUDA or CPU
pretrained_dir = "./checkpoints/trained_netG.pth"
netG = Generator().to(device)
netG.load_state_dict(torch.load(pretrained_dir, map_location=device))
torch.cuda.empty_cache()

# Helper functions
def inv_normalize(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img

def predict_images(image_list):
    trf = get_no_aug_transform()
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list])).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            generated_images = netG(image_list)
    generated_images = inv_normalize(generated_images)

    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

def predict_file(input_path, output_path, batch_size, frame_rate, output_quality, transformations):
    if mimetypes.guess_type(input_path)[0].startswith("image"):
        try:
            image = Image.open(input_path).convert('RGB')
            predicted_image = predict_images([image])[0]
            predicted_image.save(output_path, quality=output_quality)
            return output_path  # Ensure the output path is returned for image files
        except Exception as e:
            raise RuntimeError(f"Error processing image: {e}")
    elif mimetypes.guess_type(input_path)[0].startswith("video"):
        temp_dir = tempfile.TemporaryDirectory()
        subprocess.run([
            "ffmpeg", "-i", input_path, "-loglevel", "error", "-stats",
            os.path.join(temp_dir.name, 'frame_%07d.png')
        ])
        frame_paths = listdir_fullpath(temp_dir.name)
        batches = [*divide_chunks(frame_paths, batch_size)]
        for path_chunk in tqdm(batches):
            imgs = [Image.open(p) for p in path_chunk]
            imgs = predict_images(imgs)
            for path, img in zip(path_chunk, imgs):
                img.save(path)
        
        # Check if the input video contains an audio stream
        ffmpeg_command = [
            "ffmpeg", "-y", "-r", str(frame_rate), "-i",
            os.path.join(temp_dir.name, 'frame_%07d.png'), "-i", input_path,
            "-map", "0:v", "-c:v", "libx264", "-q:v", str(output_quality)
        ]
        
        # If audio exists in the input, map it to the output
        audio_check = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", input_path],
            capture_output=True, text=True
        )
        
        if audio_check.stdout.strip():  # Audio stream exists
            ffmpeg_command.extend(["-map", "1:a", "-c:a", "aac"])
        
        output_video_path = os.path.join(OUTPUT_FOLDER, f"processed_{os.path.basename(input_path)}.mp4")
        subprocess.run(ffmpeg_command + [output_video_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return output_video_path
    else:
        raise ValueError("Unsupported file type")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the input file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")
    file.save(input_path)

    # Retrieve parameters
    frame_rate = int(request.form.get('frame_rate', 30))
    batch_size = int(request.form.get('batch_size', 1))
    checkpoint = request.form.get('model_checkpoint', './checkpoints/trained_netG.pth')
    output_quality = int(request.form.get('output_quality', 90))
    device = request.form.get('device', 'cuda')
    transformations = request.form.get('transformations', 'no_augment')

    # Update device and load the model
    device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    netG.load_state_dict(torch.load(checkpoint, map_location=device))

    # Process file
    try:
        output_file_path = predict_file(input_path, output_path, batch_size, frame_rate, output_quality, transformations)
        print(f"Output file path: {output_file_path}")
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": str(e)}), 500

    # Ensure output file exists
    if not output_file_path or not os.path.exists(output_file_path):
        print(f"Error: Output file is None or not found at {output_file_path}")
        return jsonify({"error": "File processing failed or file not found."}), 404

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(output_file_path)
    print(f"Detected MIME type: {mime_type}")

    if mime_type:
        if mime_type.startswith('image'):
            print("Detected image")
            return jsonify({
                "file_url": f"/output/{os.path.basename(output_file_path)}",
                "file_type": "image"
            })
        elif mime_type.startswith('video'):
            print("Detected video")
            return jsonify({
                "file_url": f"/output/{os.path.basename(output_file_path)}",
                "file_type": "video"
            })
        else:
            print("Unsupported file type processed.")
            return jsonify({"error": "Unsupported file type processed"}), 400
    else:
        print("Mime type detection failed")
        return jsonify({"error": "Mime type detection failed"}), 400




# Serve the processed file
@app.route('/output/<filename>')
def serve_output(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({"error": "File not found"}), 404
if __name__ == '__main__':
    app.run(debug=True)
