import argparse
import os
import csv
import base64
import uuid
from PIL import Image
from io import BytesIO
import tqdm

from mcli import predict, get_inference_deployment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=str, required=True, help='deployment name to use for images')
    parser.add_argument('--prompts', type=str, required=True, help='Prompts to use')
    parser.add_argument('-s', '--seed', type=int, default=17, help='Starting random seed. will be incremented for each image')
    parser.add_argument('-g', '--guidance-scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--size', type=int, default=256, help='Image size to generate')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    args = parser.parse_args()
    return args


def fetch_image_from_deployment(deployment, prompt, seed, guidance_scale, size):
    img = predict(deployment, {"inputs": [prompt],
                            "parameters": {"height": size, "width": size, "seed": seed, "guidance_scale": guidance_scale}})
    img_data = base64.b64decode(img['output'][0])
    img = Image.open(BytesIO(img_data))
    return img


def main():
    args = parse_args()
    # Check if the output directory exists, if not create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Get the prompts from tsv file
    with open(args.prompts, 'r') as f:
        rows = list(csv.reader(f, delimiter='\t'))
        header = rows[0]
        prompts = rows[1:]

    fieldnames = ['prompt_id', 'guidance_scale', 'seed', 'text', 'category', 'challenge', 'note', 'img_path']
    metadata_filename = os.path.join(args.output, 'metadata.tsv')
    # If the file is empty, write the header.
    if os.stat(metadata_filename).st_size == 0:
        with open(metadata_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    # Otherwise, read the existing metadata
    else:
        with open(metadata_filename, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            metadata = [row for row in reader] 
            # Get list of prompts that have already been generated
            generated_prompts = [int(row['prompt_id']) for row in metadata]

    with open(metadata_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Generate the images
        for i, prompt in tqdm.tqdm(enumerate(prompts)):
            if i not in generated_prompts:
                text = prompt[0]
                category = prompt[1]
                challenge = prompt[2]
                note = prompt[3]
                # Make an image
                seed = args.seed + i
                img = fetch_image_from_deployment(args.deployment, text, seed, args.guidance_scale, args.size)
                img_id = uuid.uuid4()
                img_path = os.path.join(args.output, category, f'{img_id}.png')
                # Save the image to a file
                if not os.path.exists(os.path.join(args.output, category)):
                    os.makedirs(os.path.join(args.output, category))
                img.save(img_path)
                # Write the metadata to a file
                data = {'prompt_id': i, 'guidance_scale': args.guidance_scale, 'seed': seed, 'text': text, 'category': category, 'challenge': challenge, 'note': note, 'img_path': img_path}
                writer.writerow(data)

if __name__ == '__main__':
    main()