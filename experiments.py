import subprocess
import random
import os
import csv
import torch

def create_csv_if_not_exists():
    csv_path = 'prompts/imagenette.csv'
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'evaluation_seed', 'case_number'])
            # Add sample data
            writer.writerow(['A photo of a cat', 42, 1])
            writer.writerow(['A photo of a dog', 43, 2])
            writer.writerow(['A photo of a car', 44, 3])

def get_device():
    return '0' if torch.cuda.is_available() else '0'

label_to_int = {
    'cat': 0,
    'dog': 1,
    'car': 2,
    'house': 3,
    'landscape': 4,
    'portrait': 5,
    'abstract': 6,
    'fantasy': 7,
    'sci-fi': 8,
    'nature': 9
}

def run_experiment(label):
    device = get_device()
    label_int = label_to_int[label]

    # First command
    command1 = [
        'python', 'train-scripts/generate_mask.py',
        '--ckpt_path', 'models/ldm/sd-v1-4-full-ema.ckpt',
        '--classes', str(label_int),
        '--device', device
    ]
    result1 = subprocess.run(command1, capture_output=True, text=True)
    print(result1.stdout)
    print(result1.stderr)
    breakpoint()
    # Second command
    command2 = [
        'python', 'train-scripts/random_label.py',
        '--train_method', 'full',
        '--alpha', '0.5',
        '--lr', '1e-5',
        '--epochs', '5',
        '--class_to_forget', str(label_int),
        '--mask_path', f'mask/{label_int}/with_0.5.pt',
        '--device', device
    ]
    result2 = subprocess.run(command2, capture_output=True, text=True)
    print(result2.stdout)
    print(result2.stderr)

    # Third command
    model_name = f'compvis-{label_int}/diffusers-{label_int}.pt'
    images_path = f'evaluation_folder/{label_int}'
    command3 = [
        'python', 'eval-scripts/generate-images.py',
        '--prompts_path', 'prompts/imagenette.csv',
        '--save_path', images_path,
        '--model_name', model_name,
        '--device', device
    ]
    result3 = subprocess.run(command3, capture_output=True, text=True)
    print(result3.stdout)
    print(result3.stderr)

    # Fourth command: Compute FID
    command4 = [
        'python', 'eval-scripts/compute-fid.py',
        '--folder_path', images_path
    ]
    result4 = subprocess.run(command4, capture_output=True, text=True)
    print(result4.stdout)
    print(result4.stderr)

    # Fifth command: Accuracy evaluation
    command5 = [
        'python', 'eval-scripts/imageclassify.py',
        '--prompts_path', 'prompts/imagenette.csv',
        '--folder_path', images_path
    ]
    result5 = subprocess.run(command5, capture_output=True, text=True)
    print(result5.stdout)
    print(result5.stderr)

def run_nsfw_experiment():
    device = get_device()

    # First command for NSFW mask generation
    command1 = [
        'python', 'train-scripts/generate_mask.py',
        '--ckpt_path', 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt',
        '--nsfw', 'True',
        '--device', device
    ]
    result1 = subprocess.run(command1, capture_output=True, text=True)
    print(result1.stdout)
    print(result1.stderr)

    # Second command for NSFW removal
    command2 = [
        'python', 'train-scripts/nsfw_removal.py',
        '--train_method', 'full',
        '--mask_path', 'mask/nude_0.5.pt',
        '--device', device
    ]
    result2 = subprocess.run(command2, capture_output=True, text=True)
    print(result2.stdout)
    print(result2.stderr)

    # Third command
    model_name = 'compvis-nude/diffusers-nude.pt'
    images_path = 'evaluation_folder/nude'
    command3 = [
        'python', 'eval-scripts/generate-images.py',
        '--prompts_path', 'prompts/imagenette.csv',
        '--save_path', images_path,
        '--model_name', model_name,
        '--device', device
    ]
    result3 = subprocess.run(command3, capture_output=True, text=True)
    print(result3.stdout)
    print(result3.stderr)

    # Fourth command: Compute FID
    command4 = [
        'python', 'eval-scripts/compute-fid.py',
        '--folder_path', images_path
    ]
    result4 = subprocess.run(command4, capture_output=True, text=True)
    print(result4.stdout)
    print(result4.stderr)

    # Fifth command: Accuracy evaluation
    command5 = [
        'python', 'eval-scripts/imageclassify.py',
        '--prompts_path', 'prompts/imagenette.csv',
        '--folder_path', images_path
    ]
    result5 = subprocess.run(command5, capture_output=True, text=True)
    print(result5.stdout)
    print(result5.stderr)

if __name__ == "__main__":
    create_csv_if_not_exists()
    
    labels = ['cat', 'dog', 'car', 'house', 'landscape', 'portrait', 'abstract', 'fantasy', 'sci-fi', 'nature']
    selected_labels = random.sample(labels, 3)
    
    for label in selected_labels:
        run_experiment(label)
    
    # Run NSFW experiment
    run_nsfw_experiment()