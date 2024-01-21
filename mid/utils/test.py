

def main():
    path = '/root/autodl-tmp/data/mid/stage2/semi_train.csv'
    with open(path) as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    text_len_list = []
    summary_len_list = []
    clinical_len_list = []
    for line in lines:
        id, text, summary, clinical = line.split(',')
        # id, text, summary = line.split(',')
        text_len = len(text.split(' '))
        summary_len = len(summary.split(' '))
        clinical_len = len(clinical.split(' '))
        text_len_list.append(text_len)
        summary_len_list.append(summary_len)
        clinical_len_list.append(clinical_len)
    print(sum(text_len_list)//len(text_len_list))
    print(sum(summary_len_list)//len(summary_len_list))
    print(sum(clinical_len_list)//len(clinical_len_list))
    print(max(text_len_list))
    print(max(summary_len_list))
    print(max(clinical_len_list))


if __name__ == "__main__":
    main()