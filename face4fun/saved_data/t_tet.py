import csv
import json

response = []
modify_path = './SAGAN_trustworthy/trialdata.csv'

with open(modify_path) as f:
    rows = csv.reader(f)
    for row in rows:
        response.append(row[0]+row[3])

l = []
dict_list = []
high = 0
low = 0

l.append(["worker_id", "assignmend id", "img1", "img2", "response", "task_type", "im1", "im2", "rt", "rep", "pair_ind"])

for r in response:
    row = []
    d = {}
    worker_id = r[0:r.find(":")]
    assign_id = r[r.find(":") + 1:r.find("{")]
    r = r[r.find("{"):]
    parsed = json.loads(r)
    if (parsed["phase"] == "TEST"):
        row.append(worker_id)
        row.append(assign_id)
        row.append(parsed["im1"])
        row.append(parsed["im2"])
        row.append(parsed["hit"])
        row.append(parsed["tasktype"])
        row.append(parsed["im1relation"])
        row.append(parsed["im2relation"])
        row.append(parsed["rt"])
        row.append(parsed["rep"])
        row.append(parsed["pair_ind"])

        d["assign_id"] = assign_id
        d["worker_id"] = worker_id
        d["im1"] = parsed["im1"]
        d["im2"] = parsed["im2"]
        d["hit"] = parsed["hit"]
        d["tasktype"] = parsed["tasktype"]
        d["im1relation"] = parsed["im1relation"]
        d["im2relation"] = parsed["im2relation"]
        d["rt"] = parsed["rt"]
        d["rep"] = parsed["rep"]
        d["pair_ind"] = parsed["pair_ind"]

        if (parsed["hit"] == True):
            high = high + 1
        else:
            low = low + 1

    total = high + low

    l.append(row)
    if d:
        dict_list.append(d)
