import json
input_path = 'train.json'
output_path = 'target.txt'
with open(input_path, 'r', encoding='utf-8') as inputf, open(output_path, 'w', encoding='utf-8') as outf:
    lines = inputf.readlines()
    for line in lines:
        line = json.loads(line)
        target = line['target']
        outf.write(target + '\n')