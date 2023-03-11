import os

import diffusers.utils
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from ldm.util import instantiate_from_config
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
# from ControlNet.annotator.canny import CannyDetector
# from ControlNet.annotator.mlsd import MLSDdetector
# from ControlNet.annotator.hed import HEDdetector, nms
# from ControlNet.annotator.openpose import OpenposeDetector
# from ControlNet.annotator.uniformer import UniformerDetector
# from ControlNet.annotator.midas import MidasDetector
from PIL import Image
import torch
import numpy as np
import uuid
import einops
from pytorch_lightning import seed_everything
import cv2
import random

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

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
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

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
        return image_mask.resize(image.size)

class ImageEditing:
    def __init__(self, device):
        print("Initializing StableDiffusionInpaint to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.inpainting = StableDiffusionInpaintPipeline.from_pretrained(        "runwayml/stable-diffusion-inpainting",).to(device)

    def remove_part_of_image(self, input):
        image_path, to_be_removed_txt = input.split(",")
        print(f'remove_part_of_image: to_be_removed {to_be_removed_txt}')
        return self.replace_part_of_image(f"{image_path},{to_be_removed_txt},background")

    def replace_part_of_image(self, input):
        image_path, to_be_replaced_txt, replace_with_txt = input.split(",")
        print(f'replace_part_of_image: replace_with_txt {replace_with_txt}')
        original_image = Image.open(image_path)
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpainting(prompt=replace_with_txt, image=original_image, mask_image=mask_image).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
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
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model, tokenizer=self.text_refine_tokenizer, device=self.device)
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
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions

class image2canny_new:
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
        canny = 255 - canny
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        return updated_image_path

class canny2image_new:
    def __init__(self, device):
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-canny"
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.image_resolution = 512
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting canny2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        image = 255 - image
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        img = Image.fromarray(img)

        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = prompt + ', ' + self.a_prompt
        image = self.pipe(prompt, img, num_inference_steps=self.num_inference_steps, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=self.unconditional_guidance_scale).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        image.save(updated_image_path)
        return updated_image_path


# class image2canny:
#     def __init__(self):
#         print("Direct detect canny.")
#         self.detector = CannyDetector()
#         self.low_thresh = 100
#         self.high_thresh = 200
#
#     def inference(self, inputs):
#         print("===>Starting image2canny Inference")
#         image = Image.open(inputs)
#         image = np.array(image)
#         canny = self.detector(image, self.low_thresh, self.high_thresh)
#         canny = 255 - canny
#         image = Image.fromarray(canny)
#         updated_image_path = get_new_image_name(inputs, func_name="edge")
#         image.save(updated_image_path)
#         return updated_image_path
#
# class canny2image:
#     def __init__(self, device):
#         print("Initialize the canny2image model.")
#         model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
#         model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_canny.pth', location='cpu'))
#         self.model = model.to(device)
#         self.device = device
#         self.ddim_sampler = DDIMSampler(self.model)
#         self.ddim_steps = 20
#         self.image_resolution = 512
#         self.num_samples = 1
#         self.save_memory = False
#         self.strength = 1.0
#         self.guess_mode = False
#         self.scale = 9.0
#         self.seed = -1
#         self.a_prompt = 'best quality, extremely detailed'
#         self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
#
#     def inference(self, inputs):
#         print("===>Starting canny2image Inference")
#         image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
#         image = Image.open(image_path)
#         image = np.array(image)
#         image = 255 - image
#         prompt = instruct_text
#         img = resize_image(HWC3(image), self.image_resolution)
#         H, W, C = img.shape
#         control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
#         control = torch.stack([control for _ in range(self.num_samples)], dim=0)
#         control = einops.rearrange(control, 'b h w c -> b c h w').clone()
#         self.seed = random.randint(0, 65535)
#         seed_everything(self.seed)
#         if self.save_memory:
#             self.model.low_vram_shift(is_diffusing=False)
#         cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
#         un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
#         shape = (4, H // 8, W // 8)
#         self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
#         samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
#         if self.save_memory:
#             self.model.low_vram_shift(is_diffusing=False)
#         x_samples = self.model.decode_first_stage(samples)
#         x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
#         updated_image_path = get_new_image_name(image_path, func_name="canny2image")
#         real_image = Image.fromarray(x_samples[0])  # get default the index0 image
#         real_image.save(updated_image_path)
#         return updated_image_path
class image2line_new:
    def __init__(self):
        self.detector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        self.value_thresh = 0.1
        self.dis_thresh = 0.1
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2line Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        mlsd = self.detector(resize_image(image, self.resolution), thr_v=self.value_thresh, thr_d=self.dis_thresh)
        mlsd = np.array(mlsd)
        mlsd = 255 - mlsd
        mlsd = Image.fromarray(mlsd)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        mlsd.save(updated_image_path)
        return updated_image_path

class line2image_new:
    def __init__(self, device):
        print("Initialize the line2image model...")
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-mlsd"
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.image_resolution = 512
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting line2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        image = 255 - image
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        img = Image.fromarray(img)

        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)

        prompt = prompt + ', ' + self.a_prompt
        image = self.pipe(prompt, img, num_inference_steps=self.num_inference_steps, eta=0.0, negative_prompt=self.n_prompt, guidance_scale=self.unconditional_guidance_scale).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="line2image")
        image.save(updated_image_path)
        return updated_image_path


class image2line:
    def __init__(self):
        print("Direct detect straight line...")
        self.detector = MLSDdetector()
        self.value_thresh = 0.1
        self.dis_thresh = 0.1
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2hough Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        hough = self.detector(resize_image(image, self.resolution), self.value_thresh, self.dis_thresh)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        hough = 255 - cv2.dilate(hough, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
        image = Image.fromarray(hough)
        image.save(updated_image_path)
        return updated_image_path


class line2image:
    def __init__(self, device):
        print("Initialize the line2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_mlsd.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting line2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        image = 255 - image
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).\
            cpu().numpy().clip(0,255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="line2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path


class image2hed:
    def __init__(self):
        print("Direct detect soft HED boundary...")
        self.detector = HEDdetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2hed Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        hed = self.detector(resize_image(image, self.resolution))
        updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        image = Image.fromarray(hed)
        image.save(updated_image_path)
        return updated_image_path


class hed2image:
    def __init__(self, device):
        print("Initialize the hed2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_hed.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting hed2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2scribble:
    def __init__(self):
        print("Direct detect scribble.")
        self.detector = HEDdetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2scribble Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
        detected_map = 255 - detected_map
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class scribble2image:
    def __init__(self, device):
        print("Initialize the scribble2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_scribble.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting scribble2image Inference")
        print(f'sketch device {self.device}')
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        image = 255 - image
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2pose:
    def __init__(self):
        print("Direct human pose.")
        self.detector = OpenposeDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2pose Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class pose2image:
    def __init__(self, device):
        print("Initialize the pose2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_openpose.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting pose2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [ self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2seg:
    def __init__(self):
        print("Direct segmentations.")
        self.detector = UniformerDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2seg Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class seg2image:
    def __init__(self, device):
        print("Initialize the seg2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_seg.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting seg2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2depth:
    def __init__(self):
        print("Direct depth estimation.")
        self.detector = MidasDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2depth Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class depth2image:
    def __init__(self, device):
        print("Initialize depth2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_depth.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting depth2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [ self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2normal:
    def __init__(self):
        print("Direct normal estimation.")
        self.detector = MidasDetector()
        self.resolution = 512
        self.bg_threshold = 0.4

    def inference(self, inputs):
        print("===>Starting image2 normal Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        _, detected_map = self.detector(resize_image(image, self.resolution), bg_th=self.bg_threshold)
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class normal2image:
    def __init__(self, device):
        print("Initialize normal2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_normal.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting normal2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = image[:, :, ::-1].copy()
        img = resize_image(HWC3(img), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="normal2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class BLIPVQA:
    def __init__(self, device):
        print("Initializing BLIP VQA to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)

    def get_answer_from_question_and_image(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert('RGB')
        print(F'BLIPVQA :question :{question}')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer