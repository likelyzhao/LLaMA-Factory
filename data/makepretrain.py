import json

"""
[
  {
    "text": "Don't think you need all the bells and whistles? No problem. McKinley Heating Service Experts Heating & Air Conditioning offers basic air cleaners that work to improve the quality of the air in your home without breaking the bank. It is a low-cost solution that will ensure you and your family are living comfortably.\nIt's a good idea to understand the efficiency rate of the filters, which measures what size of molecules can get through the filter. Basic air cleaners can filter some of the dust, dander and pollen that need to be removed. They are 85% efficient, and usually have a 6-inch cleaning surface.\nBasic air cleaners are not too expensive and do the job well. If you do want to hear more about upgrading from a basic air cleaner, let the NATE-certified experts at McKinley Heating Service Experts in Edmonton talk to you about their selection.\nEither way, now's a perfect time to enhance and protect the indoor air quality in your home, for you and your loved ones.\nIf you want expert advice and quality service in Edmonton, give McKinley Heating Service Experts a call at 780-800-7092 to get your questions or concerns related to your HVAC system addressed."
  }
]
"""

totalpt = []

count =0
total = 0
with open("pt.txt") as f:
    for line in f.readlines():
        total +=1
        if len(line) > 4096:
            count+=1
            continue
        newinfo = {}
        newinfo["text"] = line
        #newinfo["output"] = [ dic[int(t)] for t in info['label']]
        totalpt.append(newinfo)
print(count)
print(total)
import random 
random.shuffle(totalpt)

with open("electric_pt_shuffle.json", "w") as f:
    json.dump(totalpt,f, ensure_ascii=False)

