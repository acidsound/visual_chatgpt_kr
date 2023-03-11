VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and image. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is  loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if not exists.
You will remember to provide the image file name loyally if it's provided in the last tool  observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

import subprocess

def execute_cmd(cmd):
    output = subprocess.check_output(cmd, shell=True)
    return output

execute_cmd('ln -s ControlNet/ldm ./ldm')
execute_cmd('ln -s ControlNet/cldm ./cldm')
execute_cmd('ln -s ControlNet/annotator ./annotator')
print(execute_cmd('nvidia-smi'))
print(execute_cmd('nvcc -V'))

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from visual_foundation_models import *
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.vectorstores import Weaviate
import re
import gradio as gr

try:
    os.mkdir('./image')
except OSError as error:
    print(error)

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)

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

def create_model(config_path, device):
    config = OmegaConf.load(config_path)
    OmegaConf.update(config, "model.params.cond_stage_config.params.device", device)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

class ConversationBot:
    def __init__(self):
        print("Initializing VisualChatGPT")
        self.llm = OpenAI(temperature=0, openai_api_key="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")
        self.edit = ImageEditing(device="cuda:0")
        self.i2t = ImageCaptioning(device="cuda:0")
        self.t2i = T2I(device="cuda:0")
        self.image2canny = image2canny_new()
        self.canny2image = canny2image_new(device="cuda:0")
        self.image2line = image2line_new()
        self.line2image = line2image_new(device="cuda:0")
        self.image2hed = image2hed_new()
        self.hed2image = hed2image_new(device="cuda:0")
        self.image2scribble = image2scribble_new()
        self.scribble2image = scribble2image_new(device="cuda:0")
        self.image2pose = image2pose_new()
        self.pose2image = pose2image_new(device="cuda:0")
        self.BLIPVQA = BLIPVQA(device="cuda:0")
        self.image2seg = image2seg_new()
        self.seg2image = seg2image_new(device="cuda:0")
        self.image2depth = image2depth_new()
        self.depth2image = depth2image_new(device="cuda:0")
        self.image2normal = image2normal_new()
        self.normal2image = normal2image_new(device="cuda:0")
        self.pix2pix = Pix2Pix(device="cuda:0")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            Tool(name="Get Photo Description", func=self.i2t.inference,
                 description="useful for when you want to know what is inside the photo. receives image_path as input. "
                             "The input to this tool should be a string, representing the image_path. "),
            Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
                 description="useful for when you want to generate an image from a user input text and it saved it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
                             "The input to this tool should be a string, representing the text used to generate image. "),
            Tool(name="Remove Something From The Photo", func=self.edit.remove_part_of_image,
                 description="useful for when you want to remove and object or something from the photo from its description or location. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the object need to be removed. "),
            Tool(name="Replace Something From The Photo", func=self.edit.replace_part_of_image,
                 description="useful for when you want to replace an object from the object description or location with another object from its description. "
                             "The input to this tool should be a comma seperated string of three, representing the image_path, the object to be replaced, the object to be replaced with "),

            # Tool(name="Instruct Image Using Text", func=self.pix2pix.inference,
            #      description="useful for when you want to the style of the image to be like the text. like: make it look like a painting. or make it like a robot. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the text. "),
            # Tool(name="Answer Question About The Image", func=self.BLIPVQA.get_answer_from_question_and_image,
            #      description="useful for when you need an answer for a question based on an image. like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the question"),
            # Tool(name="Edge Detection On Image", func=self.image2canny.inference,
            #      description="useful for when you want to detect the edge of the image. like: detect the edges of this image, or canny detection on image, or peform edge detection on this image, or detect the canny image of this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Canny Image", func=self.canny2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and a canny image. like: generate a real image of a object or something from this canny image, or generate a new real image of a object or something from this edge image. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description. "),
            # Tool(name="Line Detection On Image", func=self.image2line.inference,
            #      description="useful for when you want to detect the straight line of the image. like: detect the straight lines of this image, or straight line detection on image, or peform straight line detection on this image, or detect the straight line image of this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Line Image", func=self.line2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and a straight line image. like: generate a real image of a object or something from this straight line image, or generate a new real image of a object or something from this straight lines. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description. "),
            # Tool(name="Hed Detection On Image", func=self.image2hed.inference,
            #      description="useful for when you want to detect the soft hed boundary of the image. like: detect the soft hed boundary of this image, or hed boundary detection on image, or peform hed boundary detection on this image, or detect soft hed boundary image of this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Soft Hed Boundary Image", func=self.hed2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and a soft hed boundary image. like: generate a real image of a object or something from this soft hed boundary image, or generate a new real image of a object or something from this hed boundary. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description"),
            # Tool(name="Segmentation On Image", func=self.image2seg.inference,
            #      description="useful for when you want to detect segmentations of the image. like: segment this image, or generate segmentations on this image, or peform segmentation on this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Segmentations", func=self.seg2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and segmentations. like: generate a real image of a object or something from this segmentation image, or generate a new real image of a object or something from these segmentations. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description"),
            # Tool(name="Predict Depth On Image", func=self.image2depth.inference,
            #      description="useful for when you want to detect depth of the image. like: generate the depth from this image, or detect the depth map on this image, or predict the depth for this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Depth",  func=self.depth2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and depth image. like: generate a real image of a object or something from this depth image, or generate a new real image of a object or something from the depth map. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description"),
            # Tool(name="Predict Normal Map On Image", func=self.image2normal.inference,
            #      description="useful for when you want to detect norm map of the image. like: generate normal map from this image, or predict normal map of this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Normal Map", func=self.normal2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and normal map. like: generate a real image of a object or something from this normal map, or generate a new real image of a object or something from the normal map. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description"),
            # Tool(name="Sketch Detection On Image", func=self.image2scribble.inference,
            #      description="useful for when you want to generate a scribble of the image. like: generate a scribble of this image, or generate a sketch from this image, detect the sketch from this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Sketch Image", func=self.scribble2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and a scribble image or a sketch image. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description"),
            # Tool(name="Pose Detection On Image", func=self.image2pose.inference,
            #      description="useful for when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. "
            #                  "The input to this tool should be a string, representing the image_path"),
            # Tool(name="Generate Image Condition On Pose Image", func=self.pose2image.inference,
            #      description="useful for when you want to generate a new real image from both the user desciption and a human pose image. like: generate a real image of a human from this human pose image, or generate a new real image of a human from this pose. "
            #                  "The input to this tool should be a comma seperated string of two, representing the image_path and the user description")
            ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, 'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

    def run_image(self, image, state, txt):
        print("===============Running run_image =============")
        print("Inputs:", image, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.i2t.inference(image_filename)
        Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(image_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + ' ' + image_filename + ' '

bot = ConversationBot()
with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    with gr.Row():
        gr.Markdown("<h3><center>Visual ChatGPT</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.7):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            clear = gr.Button("Clear️")
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("Upload", file_types=["image"])

    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)

    btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])

    clear.click(bot.memory.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)


    demo.launch(server_name="0.0.0.0", server_port=7860)
