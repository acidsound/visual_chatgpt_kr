from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

import os
import random
import torch
import cv2
import uuid
from PIL import Image
import numpy as np
from pytorch_lightning import seed_everything

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    return os.path.join(head, new_file_name)


class MaskFormer:
    def __init__(self, device):
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", torch_dtype=torch.float16)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", torch_dtype=torch.float16).to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt",).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)

class ImageEditing:
    def __init__(self, device):
        print("Initializing StableDiffusionInpaint to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.inpainting = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to(device)

    def remove_part_of_image(self, input):
        image_path, to_be_removed_txt = input.split(",")
        print(f'remove_part_of_image: to_be_removed {to_be_removed_txt}')
        return self.replace_part_of_image(f"{image_path},{to_be_removed_txt},background")

    def replace_part_of_image(self, input):
        image_path, to_be_replaced_txt, replace_with_txt = input.split(",")
        print(f'replace_part_of_image: replace_with_txt {replace_with_txt}')
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpainting(prompt=replace_with_txt, image=original_image.resize((512,512)), mask_image=mask_image.resize((512,512))).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)
        return updated_image_path

class Pix2Pix:
    def __init__(self, device):
        print("Initializing Pix2Pix to %s" % device)
        self.device = device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting Pix2Pix Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = self.pipe(instruct_text,image=original_image,num_inference_steps=40,image_guidance_scale=1.2,).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        return updated_image_path

class T2I:
    def __init__(self, device):
        print("Initializing T2I to %s" % device)
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch.float16)
        self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch.float16)
        self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model, tokenizer=self.text_refine_tokenizer, device=self.device, torch_dtype=torch.float16)
        self.pipe.to(device)

    def inference(self, text):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        refined_text = self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        print(f'{text} refined to {refined_text}')
        image = self.pipe(refined_text).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename

class ImageCaptioning:
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(self.device)

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions

class image2canny:
    def __init__(self):
        print("Direct detect canny.")
        self.low_threshold = 100
        self.high_threshold = 200

    def inference(self, inputs):
        print("===>Starting image2canny Inference")
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        return updated_image_path

class canny2image:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting canny2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        image.save(updated_image_path)
        return updated_image_path

class image2line:
    def __init__(self):
        self.detector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    def inference(self, inputs):
        print("===>Starting image2line Inference")
        image = Image.open(inputs)
        mlsd = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        mlsd.save(updated_image_path)
        return updated_image_path

class line2image:
    def __init__(self, device):
        print("Initialize the line2image model...")
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-mlsd", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting line2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="line2image")
        image.save(updated_image_path)
        return updated_image_path

class image2hed:
    def __init__(self):
        print("Direct detect soft HED boundary...")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    def inference(self, inputs):
        print("===>Starting image2hed Inference")
        image = Image.open(inputs)
        hed = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        hed.save(updated_image_path)
        return updated_image_path

class hed2image:
    def __init__(self, device):
        print("Initialize the hed2image model...")
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting hed2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        image.save(updated_image_path)
        return updated_image_path

class image2scribble:
    def __init__(self):
        print("Direct detect scribble.")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    def inference(self, inputs):
        print("===>Starting image2scribble Inference")
        image = Image.open(inputs)
        scribble = self.detector(image, scribble=True)
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        scribble.save(updated_image_path)
        return updated_image_path

class scribble2image:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-scribble", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting scribble2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        image.save(updated_image_path)
        return updated_image_path

class image2pose:
    def __init__(self):
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    def inference(self, inputs):
        print("===>Starting image2pose Inference")
        image = Image.open(inputs)
        pose = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        pose.save(updated_image_path)
        return updated_image_path

class pose2image:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting pose2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        image.save(updated_image_path)
        return updated_image_path

class image2seg:
    def __init__(self):
        print("Initialize image2segmentation Inference")
        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

        self.ade_palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                    [102, 255, 0], [92, 0, 255]]

    def inference(self, inputs):
        image = Image.open(inputs)
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(self.ade_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        segmentation = Image.fromarray(color_seg)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        segmentation.save(updated_image_path)
        return updated_image_path

class seg2image:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting seg2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        image.save(updated_image_path)
        return updated_image_path

class image2depth:
    def __init__(self):
        print("initialize depth estimation")
        self.depth_estimator = pipeline('depth-estimation')

    def inference(self, inputs):
        image = Image.open(inputs)
        depth = self.depth_estimator(image)['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        depth.save(updated_image_path)
        return updated_image_path

class depth2image:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting depth2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        image.save(updated_image_path)
        return updated_image_path

class image2normal:
    def __init__(self):
        print("normal estimation")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        self.bg_threhold = 0.4

    def inference(self, inputs):
        image = Image.open(inputs)
        original_size = image.size
        image = self.depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        image.save(updated_image_path)
        return updated_image_path

class normal2image:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting normal2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="normal2image")
        image.save(updated_image_path)
        return updated_image_path

class BLIPVQA:
    def __init__(self, device):
        print("Initializing BLIP VQA to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", torch_dtype=torch.float16)
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", torch_dtype=torch.float16).to(self.device)

    def get_answer_from_question_and_image(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert('RGB')
        print(F'BLIPVQA :question :{question}')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer