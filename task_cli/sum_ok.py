import json

path = "./out/Qwen2.5-3B-Instruct__t_q3.jsonl"
count_t = 0
sub = ""


with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("dataset") != sub:           
            print(f"数据集:{sub}")
            print(f"正确数量:{count_t}")
            sub = obj.get("dataset") 
            count_t = 0
        if obj.get("correct") == True:
            count_t += 1
    


# import json, itertools
# bad = []
# with open("work/pred_qwen15b.jsonl","r",encoding="utf-8") as f:
#     for line in f:
#         r = json.loads(line)
#         if not r.get("ok"):
#             bad.append(r)
#         if len(bad) >= 10:
#             break

# for r in bad:
#     print(r["id"], r["dataset"], "| raw:", r["raw"][:160].replace("\n"," "), "| reason:", r.get("reason"))
