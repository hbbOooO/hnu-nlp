import json


def clip():
    path = '/root/autodl-tmp/data/mid/train.csv'
    with open(path, encoding='utf-8') as f:
        data = f.readlines()
    data_len = len(data)
    train_data = data[:data_len-2000]
    val_data = data[data_len-2000:]
    # test_data = data[int(0.9*data_len):]

    out_dir = '/root/autodl-tmp/data/mid/'
    with open(out_dir+'clip_train.csv', mode='w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(out_dir+'clip_val.csv', mode='w', encoding='utf-8') as f:
        f.writelines(val_data)
    # with open(out_dir+'clip_test.json', mode='w', encoding='utf-8') as f:
    #     f.writelines(test_data)

if __name__ == "__main__":
    clip()