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

VISUAL_CHATGPT_PREFIX_KR = """Visual ChatGPT는 간단한 질문에 대한 답변부터 다양한 주제에 대한 심층적인 설명 및 토론에 이르기까지 광범위한 텍스트 및 시각 관련 작업을 지원할 수 있도록 설계되었습니다. Visual ChatGPT는 수신된 입력을 기반으로 사람과 유사한 텍스트를 생성할 수 있으므로 자연스러운 대화를 수행하고 당면한 주제에 대해 일관되고 관련성 있는 답변을 제공할 수 있습니다.

Visual ChatGPT는 대량의 텍스트와 이미지를 처리하고 이해할 수 있습니다. 언어 모델로서 Visual ChatGPT는 이미지를 직접 읽을 수는 없지만 다양한 시각적 작업을 수행할 수 있는 다양한 도구가 있습니다. 각 이미지에는 "image/xxx.png" 형식의 파일 이름이 지정되며 Visual ChatGPT는 이미지를 간접적으로 이해하기 위해 다양한 도구를 호출할 수 있습니다. 이미지에 대해 이야기 할 때 Visual ChatGPT는 파일 이름에 대해 매우 엄격하며 존재하지 않는 파일을 위조하지 않습니다. 도구를 사용하여 새 이미지 파일을 생성 할 때 Visual ChatGPT는 이미지가 사용자가 필요로하는 것과 동일하지 않을 수 있음을 인식하고 다른 시각적 퀴즈 도구 또는 설명 도구를 사용하여 실제 이미지를 살펴 봅니다. Visual ChatGPT는 도구를 순서대로 사용할 수 있으며 이미지 콘텐츠와 이미지 파일 이름을 위조하지 않고 도구 관찰 결과에 충실하게 유지합니다. 새 이미지가 생성되면 마지막 도구 관찰의 파일 이름을 제공하는 것을 기억합니다.

사용자는 설명이 포함된 새 이미지를 Visual ChatGPT에 제공할 수 있습니다. 설명은 Visual ChatGPT가 이미지를 이해하는 데 도움이 되지만, Visual ChatGPT는 설명으로 직접 이미지를 상상하기보다는 도구를 사용하여 다음 작업을 수행해야 합니다. 도구는 영어로 설명을 반환하지만 사용자와의 채팅은 한국어로 해야 합니다.

전반적으로 Visual ChatGPT는 다양한 작업에 도움을 주고 다양한 주제에 대한 귀중한 통찰력과 정보를 제공할 수 있는 강력한 시각적 대화 보조 도구입니다.

도구 목록
------

Visual ChatGPT는 다음 도구와 함께 사용할 수 있습니다:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_KR = """사용자는 한국어를 사용하여 채팅하지만 도구의 매개 변수는 영어로 되어 있어야 합니다. 도구를 호출하려면 다음 형식을 따라야 합니다:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

더 이상 도구를 계속 호출할 필요가 없고 Observation에 대한 요약을 제공하려는 경우에는 영어만 사용하는 다음 형식을 사용해야 합니다:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX_KR = """정확한 파일 이름에 대해 매우 엄격하며 존재하지 않는 파일을 위조하지 않습니다.

시작하세요!

Visual ChatGPT는 텍스트 언어 모델이기 때문에 상상력에 의존하기보다는 도구를 사용하여 이미지를 관찰해야 합니다.
아이디어와 관찰에 대한 추론은 Visual ChatGPT에서만 볼 수 있으며, 최종 답장에서 사용자에게 중요한 정보를 반복하는 것을 기억해야 하며, 한국어 문장만 사용자에게 반환할 수 있습니다. 단계별로 생각해 봅시다. 도구를 사용할 때 도구의 매개 변수는 영어로만 제공됩니다.

채팅기록:
{chat_history}

새입력: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""

from visual_foundation_models import *
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import gradio as gr
import inspect


def cut_dialogue_history(history_memory, keep_last_n_words=400):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('((image|이미지)(/\S*png))', lambda m: f'![](./file=image{m.group(3)})*image{m.group(3)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
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
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '

    def init_agent(self, openai_api_key, lang):
        self.memory.clear()
        if lang=='English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, VISUAL_CHATGPT_SUFFIX
            place = "Enter text and press enter, or upload an image"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX_KR, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_KR, VISUAL_CHATGPT_SUFFIX_KR
            place = "텍스트를 입력하거나 이미지를 업로드합니다."
            label_clear = "삭제"
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, )

        return gr.update(visible = True)

bot = ConversationBot({'Text2Image': 'cuda:0',
                       'ImageCaptioning': 'cuda:0',
                       'ImageEditing': 'cuda:0',
                       'VisualQuestionAnswering': 'cuda:0',
                       'Image2Canny': 'cpu',
                       'CannyText2Image': 'cuda:0',
                       'InstructPix2Pix': 'cuda:0',
                       'Image2Depth': 'cpu',
                       'DepthText2Image': 'cuda:0',
                       })

with gr.Blocks(css="#chatbot {overflow:auto; height:500px;}") as demo:
    gr.Markdown("<h3><center>Visual ChatGPT</center></h3>")
    gr.Markdown(
        """This is a demo to the work [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models](https://github.com/microsoft/visual-chatgpt).<br>
        This space connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting.<br>  
        """
    )

    with gr.Row():
        lang = gr.Radio(choices=['Korean', 'English'], value='English', label='Language')
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here to start Visual ChatGPT(sk-...) and press Enter ↵️",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
    state = gr.State([])

    with gr.Row(visible=False) as input_raws:
        with gr.Column(scale=0.7):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
        with gr.Column(scale=0.10, min_width=0):
            run = gr.Button("🏃‍♂️Run")
        with gr.Column(scale=0.10, min_width=0):
            clear = gr.Button("🔄Clear️")
        with gr.Column(scale=0.10, min_width=0):
            btn = gr.UploadButton("🖼️Upload", file_types=["image"])

    gr.Examples(
        examples=["Generate a figure of a cat running in the garden",
                  "Replace the cat with a dog",
                  "Remove the dog in this image",
                  "Can you detect the canny edge of this image?",
                  "Can you use this canny image to generate an oil painting of a dog",
                  "Make it like water-color painting",
                  "What is the background color",
                  "Describe this image",
                  "please detect the depth of this image",
                  "Can you use this depth image to generate a cute dog",
                  ],
        inputs=txt
    )

    gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:
                <a href="https://huggingface.co/spaces/microsoft/visual_chatgpt?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br>
            </center>''')

    openai_api_key_textbox.submit(bot.init_agent, [openai_api_key_textbox, lang], [input_raws])
    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)
    run.click(bot.run_text, [txt, state], [chatbot, state])
    run.click(lambda: "", None, txt)
    btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
    clear.click(bot.memory.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)

demo.queue(concurrency_count=10).launch(server_name="0.0.0.0", server_port=47860)
