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
    # 递归搜 *.jsonl，偏重 validation/dev/test
    cands = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith('.jsonl') and re.search(r'(validation|dev|test)', fn, re.I):
                cands.append(os.path.join(dirpath, fn))
    return sorted(cands)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation')
    ap.add_argument('--ms-root', default=None, help='ModelScope 本地目录（优先使用）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    if args.ms_root:
        for fp in find_files_ms(args.ms_root):
            for ex in iter_jsonl(fp):
                ctx = ex.get('ctx') or ex.get('context') or ''
                ends = ex.get('endings') or ex.get('options') or []
                prompt = ctx + "\nOptions:\n" + "\n".join(f"({chr(65+j)}) {e}" for j, e in enumerate(ends))
                rid = ex.get('ind') or ex.get('id') or os.path.basename(fp)
                rows.append({
                    'id': str(rid),
                    'prompt': prompt,
                    'meta': {'domain':'commonsense','difficulty':'medium','metrics':{'metric':'acc','goal':'max'}}
                })
    else:
        if load_dataset is None:
            raise RuntimeError('datasets 未安装，且未提供 --ms-root')
        ds = load_dataset('hellaswag')
        split = ds[args.split]
        for i, ex in enumerate(split):
            ctx = ex.get('ctx',''); ends = ex.get('endings', [])
            prompt = ctx + "\nOptions:\n" + "\n".join(f"({chr(65+j)}) {e}" for j, e in enumerate(ends))
            rid = ex.get('ind') or f'hs-{i:06d}'
            rows.append({
                'id': str(rid),
                'prompt': prompt,
                'meta': {'domain':'commonsense','difficulty':'medium','metrics':{'metric':'acc','goal':'max'}}
            })

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('[done]', args.out, 'n=', len(rows))

if __name__ == '__main__':
    main()
