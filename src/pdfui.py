import gradio as gr
from gradio_pdf import PDF
import fitz
from llmtuner import ChatModel

def loadpdf(path):
    print(path)
    doc = fitz.open(path)
    txt = ""
    for page in doc:
        text = page.get_text()
        txt += text + "\n"
    
    return txt

def  makequestion(content, evt: gr.SelectData):
    # print(content)
    # print(evt.value)
    return evt.value

def  makeselection(content, evt: gr.SelectData):
    # print(content)
    # print(evt.value)
    return evt.value

args = {}
args['model_name_or_path'] = "/workspace/mnt/storage/zhaozhijian/yara/LLMModels/Qwen1___5-7B-Chat/"
args['template'] = "qwen"
args['prompt_path'] = "/workspace/mnt/storage/zhaozhijian/yara/LLaMA-Factory/template/QAtemplat.txt"
with open(args['prompt_path'], 'r') as f:
    template = ''.join(f.readlines())

model = ChatModel(args)

def add_text(history, task_history, query):
    def formatcontent(prompt:str, query: str) -> str:
        qa_num = 3
        p = template.format(count=qa_num, text=query)
        return p
    history += [(formatcontent(template, query), None)]
    task_history += [task_history + [("111", None)]]
    #history = history + [(_parse_text(text), None)]
    #task_history = task_history + [(task_text, None)]
    print(history)
    return history, task_history, ""




def makeQA(_chatbot, task_history):
   #import pdb
    #pdb.set_trace()
    chat_query = _chatbot[-1][0]
    query = task_history[-1][0]
    print("User: " + chat_query)

    message = {"role": "user", "content": chat_query}
    respones = ""
    for new_text in model.stream_chat([message]):
        respones += new_text
        print(respones)
        _chatbot[-1] = (chat_query, respones)
        #print(_chatbot)
        #_chatbot += (p, new_text)
        yield _chatbot
    return _chatbot

def reset_user_input():
    return gr.update(value="")


def reset_state(task_history):
    task_history.clear()
    return []



with gr.Blocks() as demo:
    with gr.Row():
        pdf = PDF(label="Upload a PDF", interactive=True)
        with gr.Column():
            name=gr.Textbox(label="info")
            content = gr.Textbox("出题的内容")
    with gr.Row():
        run = gr.Button("Make Question")
        empty_bin = gr.Button("Clear")
    name.select(makeselection, name, content)
    pdf.upload(loadpdf, pdf, name) 
    chatbot = gr.Chatbot(label="QA")
    task_history = gr.State([])
    run.click(add_text, [chatbot, task_history, content], [chatbot, task_history]).then(
            makeQA, [chatbot, task_history], [chatbot], show_progress=True
        )
    run.click(reset_user_input, [], [content])
    #run.click(reset_user_input, [], [chatbot])
    empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
    #run.click(makeQA, [chatbot, content], [chatbot], show_progress=True)

demo.launch(server_name="0.0.0.0" ,server_port=8501)