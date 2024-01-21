

def trans():
    with open(path_a) as f:
        data_a = f.readlines()
    data_a = [item[:-1] for item in data_a]
    data_a_dict = {}
    for item in data_a:
        id, text = item.split(',')
        data_a_dict[id] = text
    with open(pred_path_a) as f:
        pred_a = f.readlines()
    pred_a = [item[:-1] for item in pred_a]
    label_a = []
    for item in pred_a:
        id, pred = item.split(',')
        text = data_a_dict[id]
        line = id + ',' + text + ',' + pred + '\n'
        label_a.append(line)
    with open(out_a, mode='w', encoding='utf-8') as f:
        f.writelines(label_a)

    with open(path_b) as f:
        data_b = f.readlines()
    data_b = [item[:-1] for item in data_b]
    data_b_dict = {}
    for item in data_b:
        id, text = item.split(',')
        data_b_dict[id] = text
    with open(pred_path_b) as f:
        pred_b = f.readlines()
    pred_b = [item[:-1] for item in pred_b]
    label_b = []
    for item in pred_b:
        id, pred = item.split(',')
        text = data_b_dict[id]
        line = id + ',' + text + ',' + pred + '\n'
        label_b.append(line)
    with open(out_b, mode='w', encoding='utf-8') as f:
        f.writelines(label_b)
    print(123)



if __name__ == "__main__":
    path_a = '/root/autodl-tmp/data/mid/preliminary_a_test.csv'
    pred_path_a = '/root/autodl-tmp/save/hnu-nlp/mid/base_large/stage2/add/out/prediction_a.csv'
    out_a = '/root/autodl-tmp/data/mid/test_a_pred_label.csv'

    path_b = '/root/autodl-tmp/data/mid/preliminary_b_test.csv'
    pred_path_b = '/root/autodl-tmp/save/hnu-nlp/mid/base_large/stage2/add/out/prediction_b.csv'
    out_b = '/root/autodl-tmp/data/mid/test_b_pred_label.csv'
    trans()


