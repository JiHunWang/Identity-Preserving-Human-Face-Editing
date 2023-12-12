import argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import (CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor)

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

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat)
        return sim_direction

    def forward(self, image_one, image_two, caption):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat = self.encode_text(caption)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat
        )
        return directional_similarity


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_id = 'openai/clip-vit-large-patch14'
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str)
    parser.add_argument('--edit_image_path', type=str)
    parser.add_argument('--caption', type=str)
    args = parser.parse_args()

    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    sim_score = dir_similarity(args.input_image_path, args.edit_image_path, args.caption)
    print(float(sim_score.detach().cpu()))