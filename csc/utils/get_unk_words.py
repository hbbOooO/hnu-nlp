import json

def get_unk_words(data_paths, vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        words = f.readlines()
    vocab_words = [w[:-1] for w in words]
    data_words = set()
    wrong_unk_item = []
    for path in data_paths:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            original_text = item['original_text']
            correct_text = item['correct_text']
            for i in range(len(original_text)):
                w = original_text[i]
                if w not in vocab_words:
                    if i in item['wrong_ids']:
                        wrong_unk_item.append(item)
                data_words.add(w)
            for w in correct_text:
                data_words.add(w)
    
    data_words = list(data_words)
    unk_words = [w for w in data_words if w not in vocab_words]
    print(unk_words)
    print(wrong_unk_item)

if __name__ == "__main__":
    data_paths = [
        '/root/autodl-tmp/data/csc/train.json',
        '/root/autodl-tmp/data/csc/dev.json',
        '/root/autodl-tmp/data/csc/test.json',]
    vocab_path = '/root/autodl-tmp/csc/my_model/macbert4csc-base-chinese/vocab.txt'
    get_unk_words(data_paths, vocab_path)