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

from visual_foundation_models import *
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import gradio as gr


def cut_dialogue_history(history_memory, keep_last_n_words=400):
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


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

        self.models = dict()
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        self.tools = []
        for class_name, instance in self.models.items():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))


    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. " \
                       "This information helps you to understand this image, " \
                       "but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(
            image_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, txt + ' ' + image_filename + ' '

    def init_agent(self, openai_api_key):
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, 'suffix': VISUAL_CHATGPT_SUFFIX}, )

        return gr.update(visible = True)

bot = ConversationBot({'Text2Image':'cuda:0',
                       'ImageCaptioning':'cuda:0',
                       'ImageEditing': 'cuda:0',
                       'VisualQuestionAnswering': 'cuda:0',
                       'Image2Canny':'cpu',
                       'CannyText2Image':'cuda:0',
                       'InstructPix2Pix':'cuda:0'})

with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    with gr.Row():
        gr.Markdown("<h3><center>Visual ChatGPT</center></h3>")

    with gr.Row():
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here to start Visual ChatGPT(sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
    state = gr.State([])

    with gr.Row(visible=False) as input_raws:
        with gr.Column(scale=0.7):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            clear = gr.Button("Clear️")
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("Upload", file_types=["image"])

    gr.Examples(
        examples=["Generate a figure of a lovely cat",
                  "Replace the cat with a lovely dog",
                  "Remove the cat in this image",
                  "Can you detect the canny edge of this image?",
                  "Can you use this canny image to generate a oil painting of a lovely dog",
                  "Make it like water-color painting",
                  "What is the background color"
                  "Describe this image"],
        inputs=txt
    )


    openai_api_key_textbox.submit(bot.init_agent, [openai_api_key_textbox], [input_raws])
    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)
    btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
    clear.click(bot.memory.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)

    demo.launch(server_name="0.0.0.0", server_port=7860)
