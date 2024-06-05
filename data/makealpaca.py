import json

"""
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
"""

dic = {
    0:"消极",
    1:'积极',
    2:'普通',
}
totalalpaca = []

count =0
total = 0

with open("sentiment_rlhf.jsonl") as f:
    for line in f.readlines():
        info = json.loads(line)
        total +=1
        if len(info["content"]) > 5000:
            count+=1
            continue
        newinfo = {}
        newinfo["instruction"] = "下面这段话中的情绪是怎么样的?请从下面三个词语中选择一个{消极/普通/积极}"
        newinfo["input"] = info["content"]
        #newinfo["output"] = [ dic[int(t)] for t in info['label']]
        totalalpaca.append(newinfo)
print(count)
print(total)

with open("sentiment_rlhf_alpace.json", "w") as f:
    json.dump(totalalpaca,f, ensure_ascii=False)

