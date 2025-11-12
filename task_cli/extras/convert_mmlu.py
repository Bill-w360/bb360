# -*- coding: utf-8 -*-
import os, re, json, argparse

try:
    from datasets import load_dataset  # 可选依赖
except Exception:
    load_dataset = None

SUBJ2DOMAIN = {
    # math
    'abstract_algebra':'math','college_mathematics':'math','high_school_mathematics':'math','high_school_statistics':'math',
    # reading-ish
    'high_school_english':'reading','reading_comprehension':'reading',
    # commonsense / soc-sci
    'high_school_psychology':'commonsense','high_school_geography':'commonsense','high_school_government_and_politics':'commonsense',
    # science
    'college_physics':'science','high_school_physics':'science','college_biology':'science','high_school_biology':'science',
    'college_chemistry':'science','high_school_chemistry':'science',
}

def map_domain(subj: str) -> str:
    return SUBJ2DOMAIN.get(subj, 'science')

def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def find_ms_subject_files(root):
    # 结构：mmlu/<subject>/{train|validation|dev|test}.jsonl
    pairs = []
    for dirpath, _, files in os.walk(root):
        subj = os.path.basename(dirpath)
        for fn in files:
            if fn.endswith('.jsonl') and re.search(r'(validation|dev|test|train)', fn, re.I):
                pairs.append((subj, os.path.join(dirpath, fn)))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation')
    ap.add_argument('--ms-root', default=None, help='ModelScope 本地目录（优先使用）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--subjects', nargs='*', default=None, help='仅转换指定子科目')
    args = ap.parse_args()

    rows = []
    if args.ms_root:
        for subj, fp in find_ms_subject_files(args.ms_root):
            if args.subjects and subj not in args.subjects:
                continue
            for ex in iter_jsonl(fp):
                q = ex.get('question','')
                choices = ex.get('choices') or [ex.get(f'choices.{k}') for k in range(4)]
                prompt = q + "\nOptions:\n" + "\n".join(
                    f"({chr(65+j)}) {e}" for j, e in enumerate(choices) if e
                )
                rid = ex.get('id') or f'mmlu-{subj}-{len(rows):06d}'
                rows.append({
                    'id': str(rid),
                    'prompt': prompt,
                    'meta': {'domain': map_domain(subj), 'difficulty':'medium',
                             'metrics': {'metric':'acc','goal':'max'}, 'subject': subj}
                })
    else:
        if load_dataset is None:
            raise RuntimeError('datasets 未安装，且未提供 --ms-root')
        ds = load_dataset('cais/mmlu', 'all')
        split = ds[args.split]
        wanted = set(args.subjects) if args.subjects else None
        for i, ex in enumerate(split):
            subj = ex.get('subject','unknown')
            if wanted and subj not in wanted:
                continue
            q = ex.get('question','')
            choices = ex.get('choices') or [ex.get(f'choices.{k}') for k in range(4)]
            prompt = q + "\nOptions:\n" + "\n".join(
                f"({chr(65+j)}) {e}" for j, e in enumerate(choices) if e
            )
            rid = ex.get('id') or f'mmlu-{subj}-{i:06d}'
            rows.append({
                'id': str(rid),
                'prompt': prompt,
                'meta': {'domain': map_domain(subj), 'difficulty':'medium',
                         'metrics': {'metric':'acc','goal':'max'}, 'subject': subj}
            })

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('[done]', args.out, 'n=', len(rows))

if __name__ == '__main__':
    main()
