# -*- coding: utf-8 -*-
import os, re, json, argparse

try:
    from datasets import load_dataset  # 可选依赖
except Exception:
    load_dataset = None

def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def find_files_ms(root):
    # 递归搜 *.jsonl，文件名里匹配 train/test/validation/main
    cands = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith('.jsonl') and re.search(r'(train|test|validation|main)', fn, re.I):
                cands.append(os.path.join(dirpath, fn))
    return sorted(cands)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='main', help='HF: main；MS: 忽略，用文件名判断')
    ap.add_argument('--ms-root', default=None, help='ModelScope 本地目录（优先使用）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    if args.ms_root:
        files = find_files_ms(args.ms_root)
        if not files:
            raise FileNotFoundError(f'No jsonl under {args.ms_root}')
        for fp in files:
            for ex in iter_jsonl(fp):
                q = ex.get('question') or ex.get('question_text') or ex.get('query') or ''
                rid = ex.get('id') or ex.get('qid') or f"{os.path.basename(fp)}:{ex.get('index', len(rows))}"
                rows.append({
                    'id': str(rid),
                    'prompt': q,
                    'meta': {'domain':'math','difficulty':'medium','metrics':{'metric':'acc','goal':'max'}}
                })
    else:
        if load_dataset is None:
            raise RuntimeError('datasets 未安装，且未提供 --ms-root')
        ds = load_dataset('gsm8k', args.split)
        split = ds['test'] if 'test' in ds else ds['train']
        for i, ex in enumerate(split):
            q = ex.get('question') or ''
            rid = ex.get('id') or f'gsm8k-{i:06d}'
            rows.append({
                'id': str(rid),
                'prompt': q,
                'meta': {'domain':'math','difficulty':'medium','metrics':{'metric':'acc','goal':'max'}}
            })

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('[done]', args.out, 'n=', len(rows))

if __name__ == '__main__':
    main()
