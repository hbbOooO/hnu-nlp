from tool.convert_from_sentpair_to_edits import convert


tgt_path = '../data/pretrain_data/target/ljp.txt'

src_path = 'my_data/1/redundant_file.txt'
output_path = 'my_data/1/redundant_output.json'
convert(src_path, tgt_path, output_path)

src_path = 'my_data/1/missing_file.txt'
output_path = 'my_data/1/missing_output.json'
convert(src_path, tgt_path, output_path)

src_path = 'my_data/1/ordering_file.txt'
output_path = 'my_data/1/ordering_output.json'
convert(src_path, tgt_path, output_path)

src_path = 'my_data/1/selection_file.txt'
output_path = 'my_data/1/selection_output.json'
convert(src_path, tgt_path, output_path)