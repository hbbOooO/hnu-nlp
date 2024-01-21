# Baseline model based on Mengzi.
#
# Author: Jingcong Liang

from pathlib import Path

import pandas as pd
import torch
from transformers import set_seed

from mengzi import make_dataset, MengziSimpleT5

from collections import defaultdict
# from pathlib import Path
from typing import DefaultDict, Dict, List

import numpy as np
# import torch
from jieba import lcut
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
# from transformers import set_seed

# from mengzi import MengziSimpleT5
from util import read_data

def eval():
    # set_seed(2022)
    data_path = Path('/root/autodl-tmp/data/ccac/track2/')
    model_path = Path('/root/autodl-tmp/model/ccac/track2/mengzi/model_finetuned/')
    test_data: Dict[str, List[str]] = read_data(data_path, 'test')

    model = MengziSimpleT5(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.from_pretrained(model_path)

    smoothing = SmoothingFunction()
    rouge = Rouge()
    scores: DefaultDict[str, List[float]] = defaultdict(list)

    for claim, references in test_data.items():
        references_tokens: List[List[str]] = [lcut(x.strip()) for x in references]

        for output in model.predict(claim, max_length=64, num_return_sequences=5, num_beams=20,
                                    repetition_penalty=5.0):
            output_tokens: List[str] = lcut(output)
            scores['bleu'].append(sentence_bleu(references_tokens, output_tokens,
                                                smoothing_function=smoothing.method1))

            for key, result in rouge.get_scores(
                [' '.join(output_tokens)] * len(references_tokens),
                [' '.join(x) for x in references_tokens], avg=True
            ).items():
                key: str
                result: Dict[str, float]
                scores[key].append(result['f'])

    for key, values in scores.items():
        score: float = np.mean(values)
        print(f'{key}: {score:.3f}')

if __name__ == '__main__':
    set_seed(2022)
    data_path = Path('/root/autodl-tmp/data/ccac/track2/')
    model_path = Path('/root/autodl-tmp/model/ccac/track2/mengzi/')

    train_set: pd.DataFrame = make_dataset(data_path, 'train')
    valid_set: pd.DataFrame = make_dataset(data_path, 'test')

    model = MengziSimpleT5(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.from_pretrained(model_path)

    model.train(train_set, valid_set, source_max_token_len=32, target_max_token_len=256, max_epochs=1, outputdir='log', dataloader_num_workers=32)
    model.save_pretrained(model_path / 'model_finetuned')

    eval()
