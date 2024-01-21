import json

def main(train_path, val_path):
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)
    data = train_data
    data.update(val_data)
    res = {}
    for claim, arguments in data.items():
        for argument, argument_cls_list in arguments.items():
            name_str = ''.join([name for name, state in argument_cls_list.items() if state == 1])
            if name_str == '未分类': continue
            if name_str not in res:
                res[name_str] = 1
            else:
                res[name_str] += 1
    res = sorted(res.items(), key=lambda x: -x[1])
    print(123)


if __name__ == "__main__":
    train_path = '/root/autodl-tmp/data/ccac/track2/angle/train_data.json'
    val_path = '/root/autodl-tmp/data/ccac/track2/angle/val_data.json'
    main(train_path, val_path)