import json


def trans(input_path, out_path):
    out_data = []
    with open(input_path) as in_f:
        in_data = in_f.readlines()
        for in_line in in_data:
            in_line = in_line[:-1]
            id, text = in_line.split(',')
            line_json = json.dumps({
                'id': id,
                'text': text
            })
            out_data.append(line_json + '\n')
    with open(out_path, 'w') as out_f:
        out_f.writelines(out_data)

if __name__ == "__main__":
    input_path = '/root/autodl-tmp/data/mid/preliminary_a_test.csv'
    out_path = '/root/autodl-tmp/data/mid/preliminary_a_test.json'
    trans(input_path, out_path)