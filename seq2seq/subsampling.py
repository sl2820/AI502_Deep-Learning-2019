import io

src_dir = '.data/wmt14/train.tok.clean.bpe.32000.de'
trg_dir = '.data/wmt14/train.tok.clean.bpe.32000.en'
save_src_dir = '.data/wmt14/train.tok.clean.bpe.32000_partial.de'
save_trg_dir = '.data/wmt14/train.tok.clean.bpe.32000_partial.en'

target_num_lines = int(4500000 * 0.01)
# target_num_lines = int(1)

count = 0
with io.open(src_dir, mode='r', encoding='utf-8') as src_file, \
        io.open(save_src_dir, mode='w', encoding='utf-8') as save_src_file,\
            io.open(trg_dir, mode='r', encoding='utf-8') as trg_file, \
                io.open(save_trg_dir, mode='w', encoding='utf-8') as save_trg_file:
    for src_line, trg_line in zip(src_file, trg_file):
        save_src_file.write(src_line)
        save_trg_file.write(trg_line)
        print(src_line)
        print(trg_line)
        count += 1
        if count >= target_num_lines:
            break
a = 1