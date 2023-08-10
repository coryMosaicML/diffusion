import gradio as gr
import os
import yaml
import base64
from PIL import Image
from io import BytesIO
import csv
import random
import time
import argparse


def load_config():
    """Load the YAML configuration file."""
    parser = argparse.ArgumentParser(description='Read a YAML config and greet.')
    parser.add_argument('config', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config


class ABTest:
    """Run an A/B test between models compared to a reference model."""

    def __init__(self, config):
        self.config = config
        self.size = config['size']
        self.fieldnames = ['Model A',
                           'Model B',
                           'Selection',
                           'Category',
                           'Challenge',
                           'Prompt',
                           'Guidance scale',
                           'Seed',
                           'Image A path',
                           'Image B path',
                           'Start time',
                           'End time',]

        # Get the metadata for the reference model
        self.reference_name = config['reference_model']['name']
        self.reference_model = self.read_metadata(config['reference_model']['path'])

        # Get the metadata for each model
        self.testing_models = {}
        for model in config['models']:
            self.testing_models[model['name']] = self.read_metadata(model['path'])

        # Check if the output file exists and write the header if it doesn't
        if not os.path.exists(config['output_file']):
            with open(config['output_file'], 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def read_metadata(self, model_path):
        with open(os.path.join(model_path, 'metadata.tsv'), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            images = [row for row in reader]

        metadata = {}
        for image in images:
            prompt_id = image['prompt_id']
            guidance_scale = image['guidance_scale']
            seed = image['seed']
            text = image['text']
            category = image['category']
            challenge = image['challenge']
            note = image['note']
            img_path = image['img_path']
            metadata[(prompt_id, guidance_scale, seed)] = {'text': text,
                                                           'guidance_scale': guidance_scale,
                                                           'seed': seed,
                                                           'category': category,
                                                           'challenge': challenge,
                                                           'note': note,
                                                           'img_path': img_path}
        return metadata
    
    def get_images(self, state):
        state = {}
        # Choose a reference image image key
        reference_key = random.choice(list(self.reference_model.keys()))
        # Store the relevant metadata in the state
        state['Category'] = self.reference_model[reference_key]['category']
        state['Challenge'] = self.reference_model[reference_key]['challenge']
        state['Prompt'] = self.reference_model[reference_key]['text']
        state['Guidance scale'] = self.reference_model[reference_key]['guidance_scale']
        state['Seed'] = self.reference_model[reference_key]['seed']
        # Load the corresponding image
        reference_img = Image.open(self.reference_model[reference_key]['img_path'])
        # Choose a random model to compare to
        model_key = random.choice(list(self.testing_models.keys()))
        model = self.testing_models[model_key]
        # Load the corresponding image
        img = Image.open(model[reference_key]['img_path'])
        # Log the starting time for the test
        state['Start time'] = time.time()
        if self.config['display_prompt']:
            display_prompt = state['Prompt']
        else:
            display_prompt = ''
        # Flip a fair coin to decide which image to show first
        if random.random() < 0.5:
            state['Model A'] = self.reference_name
            state['Model B'] = model_key
            state['Image A path'] = self.reference_model[reference_key]['img_path']
            state['Image B path'] = model[reference_key]['img_path']
            return state, reference_img, img, display_prompt
        else:
            state['Model A'] = model_key
            state['Model B'] = self.reference_name
            state['Image A path'] = model[reference_key]['img_path']
            state['Image B path'] = self.reference_model[reference_key]['img_path']
            return state, img, reference_img, display_prompt


    def log_data(self, state, selection):
        state['Selection'] = selection
        # Log the ending time for the test
        state['End time'] = time.time()
        # Write to CSV
        with open(self.config['output_file'], 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(state)
        state = {}
        return state

    def log_data_1(self, state):
        state = self.log_data(state, state['Model A'])
        state, img1, img2, prompt = self.get_images(state)
        return state, img1, img2, prompt
    
    def log_data_2(self, state):
        state = self.log_data(state, state['Model B'])
        state, img1, img2, prompt = self.get_images(state)
        return state, img1, img2, prompt
    
    def run_test(self):
        with gr.Blocks() as demo:
            state = gr.State({})
            with gr.Column():
                with gr.Row():
                    text = gr.Label(value=self.config['question'], show_label=True)
                with gr.Row():
                    generate_button = gr.Button("Generate")
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt")
                with gr.Row():
                    img1 = gr.Image(type='pil', label="Image 1").style(height=self.size, width=self.size)
                    img2 = gr.Image(type='pil', label="Image 2").style(height=self.size, width=self.size)
                with gr.Row():
                    img1_button = gr.Button("Select Image 1")
                    img2_button = gr.Button("Select Image 2")
            generate_button.click(self.get_images, inputs=[state], outputs=[state, img1, img2, prompt])
            img1_button.click(self.log_data_1, inputs=[state], outputs=[state, img1, img2, prompt])
            img2_button.click(self.log_data_2, inputs=[state], outputs=[state, img1, img2, prompt])

        demo.launch(share=self.config['share'])


if __name__ == '__main__':
    config = load_config()
    test = ABTest(config)
    test.run_test()