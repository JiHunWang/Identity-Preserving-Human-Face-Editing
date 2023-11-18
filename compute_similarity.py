import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as fu
from torchvision.transforms import functional as F
from transformers import (CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor)
from torchmetrics.image.fid import FrechetInceptionDistance

# instructpix2pix: CLIPTextModel (clip-vit-large-patch14)
#                  CLIPTokenizer
#                  CLIPImageProcessor

class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat, text_feat):
        sim_direction = fu.cosine_similarity(img_feat, text_feat)
        return sim_direction

    def forward(self, image, caption):
        img_feat = self.encode_image(image)
        text_feat = self.encode_text(caption)
        directional_similarity = self.compute_directional_similarity(
            img_feat, text_feat
        )
        return directional_similarity


def preprocess_image(path, image):
    image = Image.open(f'{path}{image}')
    image = F.pil_to_tensor(image)
    image = image.unsqueeze(0)
    image = image / 255.0
    return F.center_crop(image, (256, 256))


'''
--real_images_path PATH/TO/REAL/IMAGES
--edit_images_path PATH/TO/EDIT/IMAGES
--captions "filename1:prompt1" "filename2:prompt2" "filename3:prompt3"
'''
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_id = 'openai/clip-vit-large-patch14'
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--real_images_path', type=str)
    parser.add_argument('--edit_images_path', type=str)
    # parser.add_argument('--captions', type=str)
    parser.add_argument('--captions', metavar='N', type=str, nargs='+',
                        help='List of captions')
    args = parser.parse_args()
    captions = {c.split(':')[0]: c.split(':')[1].strip() for c in args.captions}

    clip_score = {}
    for idx, file in enumerate(os.listdir(args.edit_images_path)):
        print(f'{args.edit_images_path}{file}:', captions[file])
        mod_image = Image.open(f'{args.edit_images_path}{file}')
        dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
        sim_score = dir_similarity(mod_image, captions[file])
        clip_score[file] = float(sim_score.detach().cpu())
    print('clip score:\n', clip_score)
    
    real_images = torch.cat([preprocess_image(args.real_images_path, image) for image in os.listdir(args.real_images_path)])
    fake_images = torch.cat([preprocess_image(args.edit_images_path, image) for image in os.listdir(args.edit_images_path)])
    print('shape:', real_images.shape, fake_images.shape)

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    print('FID:', float(fid.compute()))

    