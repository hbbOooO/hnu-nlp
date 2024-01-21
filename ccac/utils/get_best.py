import os
from rouge import Rouge
from jieba import lcut
from tqdm import tqdm

def find_lcsubstr(s1, s2): 
    # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] 
    mmax = 0   # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p], mmax   # 返回最长子串及其长度

def bottom_up_dp_lcs(str_a, str_b):
  """
  longest common subsequence of str_a and str_b
  """
  if len(str_a) == 0 or len(str_b) == 0:
    return 0
  dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
  for i in range(1, len(str_a) + 1):
    for j in range(1, len(str_b) + 1):
      if str_a[i-1] == str_b[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
      else:
        dp[i][j] = max([dp[i-1][j], dp[i][j-1]])
#   print "length of LCS is :",dp[len(str_a)][len(str_b)]
  # 输出最长公共子序列
  i, j = len(str_a), len(str_b)
  LCS = ""
  while i > 0 and j > 0:
    if str_a[i-1] == str_b[j-1] and dp[i][j] == dp[i-1][j-1] + 1:
      LCS = str_a[i - 1] + LCS
      i, j = i-1, j-1
      continue
    if dp[i][j] == dp[i-1][j]:
      i, j = i-1, j
      continue
    if dp[i][j] == dp[i][j-1]:
      i, j = i, j-1
      continue
  return LCS

def get_best(data_root_dir):
    rouge = Rouge()
    filenames = os.listdir(data_root_dir)
    filenames = [name for name in filenames if '.txt' in name]
    for name in filenames[3:]:
        with open(data_root_dir + name) as f:
            arguments = f.readlines()
        arguments = [argument[:-1] for argument in arguments]
        claim = name[:-4]
        lcs_list = []
        for i in range(len(arguments)):
            for j in range(i+1, len(arguments)):
                lcs = bottom_up_dp_lcs(arguments[i], arguments[j])
                if lcs != '': lcs_list.append(lcut(lcs))
        arguments_tokens = [' '.join(lcut(argument)) for argument in arguments]
        score_list = []
        for lcs in tqdm(lcs_list):
            score = rouge.get_scores([' '.join(lcs)] * len(arguments), arguments_tokens, avg=True, ignore_empty=True)['rouge-l']['f']
            score_list.append(score)
        print(claim,
              max(score_list),
              lcs_list[score_list.index(max(score_list))])

def get_best_by_original(data_root_dir):
    rouge = Rouge()
    filenames = os.listdir(data_root_dir)
    filenames = [name for name in filenames if '.txt' in name]
    for name in filenames:
        claim = name[:-4]
        with open(data_root_dir + name) as f:
            arguments = f.readlines()
        arguments_tokens = [' '.join(lcut(argument)) for argument in arguments]
        score_list = []
        for argument in arguments:
            argument_tokens = lcut(argument)
            score = rouge.get_scores([' '.join(argument_tokens)] * len(arguments), arguments_tokens, avg=True, ignore_empty=True)['rouge-l']['f']
            score_list.append(score)
        print(claim,
              max(score_list),
              arguments[score_list.index(max(score_list))])
        

if __name__ == "__main__":
    data_root_dir = '/root/autodl-tmp/data/ccac/track2/original/'
    get_best_by_original(data_root_dir)

# print(find_lcsubstr('abfcdfg', 'abcdfg'))
