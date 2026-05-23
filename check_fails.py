import json, os, glob
files = glob.glob('data/benchmark/*.json')
latest = max(files, key=os.path.getmtime)
data = json.load(open(latest))
for r in data['results']:
    a = 'v' if r['condition_a']['correct'] else 'X'
    b = 'v' if r['condition_b']['correct'] else 'X'
    d = 'v' if r['condition_d']['correct'] else 'X'
    f = 'v' if r['condition_f']['correct'] else 'X'
    if 'X' in [a,b,d,f]:
        print(f"{r['word']:<12} A:{a} B:{b} D:{d} F:{f}  expect:{r['correct_context']}")
