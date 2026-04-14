#!/bin/bash
set -e

# KOVA AZR — MASTER BOOTSTRAP
# One script. Everything. Start to finish.
# ============================================

echo '=== KOVA AZR BOOTSTRAP ==='
echo 'This script runs the entire pipeline:'
echo '  1. Environment setup'
echo '  2. Create all source files'
echo '  3. Smoke test'
echo '  4. Dataset preparation'
echo '  5. SFT training'
echo '  6. AZR GRPO training (16 hours)'
echo '  7. Auto-benchmarks'
echo ''

# ---------- CONFIG ----------
export KOVA_ROOT="$HOME/kova"
export BASE_MODEL_ID='Qwen/Qwen3.5-9B'
export KOVA_DATA_ROOT="$KOVA_ROOT/data"
export KOVA_CKPT_ROOT="$KOVA_ROOT/checkpoints"
export KOVA_RUN_NAME='kova-azr-gemma4'
export PYTHONPATH="$KOVA_ROOT/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

GRPO_HOURS=17

mkdir -p "$KOVA_ROOT" && cd "$KOVA_ROOT"
mkdir -p src/kova/verifiers scripts data/seeds data/evals checkpoints logs

# ---------- SYSTEM PACKAGES ----------
echo ''
echo '=== Installing system packages ==='
apt-get update -qq && apt-get install -y -qq \
  build-essential git curl wget sqlite3 openjdk-17-jdk \
  rustc cargo nodejs npm gcc g++ golang-go 2>/dev/null

# ---------- PYTHON PACKAGES ----------
echo ''
echo '=== Installing Python packages ==='
pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
  sentencepiece safetensors sympy wandb playwright pytest huggingface_hub \
  lm-eval tensorboard
python -m playwright install chromium 2>/dev/null
cd "$KOVA_ROOT" && npm install --silent axe-core tsx three 2>/dev/null

# ---------- CREATE SOURCE FILES ----------
echo ''
echo '=== Creating source files ==='

# --- Executor ---
cat > src/kova/__init__.py << 'EOF'
EOF

cat > src/kova/verifiers/__init__.py << 'EOF'
EOF

cat > src/kova/executor.py << 'PYEOF'
from __future__ import annotations
import sqlite3, subprocess, tempfile
from pathlib import Path

class MultiExec:
    def __init__(self, timeout=15):
        self.timeout = timeout
    def _run(self, cmd, cwd=None, stdin=None):
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=self.timeout, input=stdin)
        return {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    def run_python(self, code): return self._run(['python', '-c', code])
    def run_bash(self, code): return self._run(['bash', '-lc', code])
    def run_node(self, code): return self._run(['node', '-e', code])
    def run_ts(self, code):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'main.tsx').write_text(code)
            return self._run(['npx', 'tsx', 'main.tsx'], cwd=d)
    def run_java(self, code):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'Main.java').write_text(code)
            r = self._run(['javac', 'Main.java'], cwd=d)
            return r if r['returncode'] != 0 else self._run(['java', 'Main'], cwd=d)
    def run_c(self, code):
        with tempfile.TemporaryDirectory() as d:
            s, o = Path(d, 'main.c'), Path(d, 'main')
            s.write_text(code)
            r = self._run(['gcc', str(s), '-o', str(o), '-lm'])
            return r if r['returncode'] != 0 else self._run([str(o)])
    def run_cpp(self, code):
        with tempfile.TemporaryDirectory() as d:
            s, o = Path(d, 'main.cpp'), Path(d, 'main')
            s.write_text(code)
            r = self._run(['g++', str(s), '-o', str(o), '-std=c++17', '-lm'])
            return r if r['returncode'] != 0 else self._run([str(o)])
    def run_rust(self, code):
        with tempfile.TemporaryDirectory() as d:
            s, o = Path(d, 'main.rs'), Path(d, 'main')
            s.write_text(code)
            r = self._run(['rustc', str(s), '-o', str(o)])
            return r if r['returncode'] != 0 else self._run([str(o)])
    def run_go(self, code):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'main.go').write_text(code)
            return self._run(['go', 'run', 'main.go'], cwd=d)
    def run_swift(self, code):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d, 'main.swift'); s.write_text(code)
            return self._run(['swift', str(s)])
    def run_sqlite(self, schema, query):
        c = sqlite3.connect(':memory:'); cur = c.cursor()
        cur.executescript(schema); rows = cur.execute(query).fetchall(); c.close()
        return {'returncode': 0, 'rows': rows}
PYEOF

# --- Verifiers ---
cat > src/kova/verifiers/math_verifier.py << 'PYEOF'
import sympy as sp
def verify_math(pred, target):
    try:
        ok = sp.simplify(pred) == sp.simplify(target)
        return {'ok': bool(ok), 'pred': pred, 'target': target}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
PYEOF

cat > src/kova/verifiers/sql_verifier.py << 'PYEOF'
from src.kova.executor import MultiExec
def verify_sql(schema, candidate, expected):
    r = MultiExec().run_sqlite(schema, candidate)
    rows = r.get('rows', [])
    return {'ok': sorted(rows) == sorted(expected), 'rows': rows, 'expected': expected}
PYEOF

cat > src/kova/verifiers/svg_verifier.py << 'PYEOF'
from xml.etree import ElementTree as ET
def verify_svg(text):
    try:
        root = ET.fromstring(text)
        ok = root.tag.endswith('svg') and any(
            c.tag.endswith(t) for c in root.iter() for t in ('path','circle','rect'))
        return {'ok': ok}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
PYEOF

cat > src/kova/verifiers/safety_classifier.py << 'PYEOF'
import re
BLOCK = [r'credential\s+theft',r'weapon',r'exfiltrat',r'ransomware',r'zero-day',r'privilege escalation']
def classify_text(text):
    low = text.lower()
    m = [p for p in BLOCK if re.search(p, low)]
    return {'allow': len(m)==0 and len(text.strip())>=8, 'matches': m}
PYEOF

cat > src/kova/safety_gate.py << 'PYEOF'
from src.kova.verifiers.safety_classifier import classify_text
def gate_example(task, sol):
    t, s = classify_text(task), classify_text(sol)
    return {'allow': t['allow'] and s['allow'], 'task_check': t, 'solution_check': s}
PYEOF

# --- Smoke test ---
cat > scripts/smoke_test.py << 'PYEOF'
import json
from src.kova.executor import MultiExec
from src.kova.verifiers.math_verifier import verify_math
from src.kova.verifiers.sql_verifier import verify_sql
from src.kova.verifiers.svg_verifier import verify_svg
from src.kova.safety_gate import gate_example

ex = MultiExec()
checks = {
    'python': ex.run_python('print(6*7)'),
    'bash': ex.run_bash('echo kova-ready'),
    'node': ex.run_node('console.log(42)'),
    'c': ex.run_c('#include <stdio.h>\nint main(){printf("42\\n");return 0;}'),
    'cpp': ex.run_cpp('#include <iostream>\nint main(){std::cout<<42<<std::endl;}'),
    'java': ex.run_java('public class Main{public static void main(String[] a){System.out.println(42);}}'),
    'go': ex.run_go('package main\nimport "fmt"\nfunc main(){fmt.Println(42)}'),
    'rust': ex.run_rust('fn main(){println!("42");}'),
    'math': verify_math('2*x+2*x', '4*x'),
    'sql': verify_sql('create table t(x int);insert into t values(1),(2);','select sum(x) from t;',[(3,)]),
    'svg': verify_svg('<svg xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10"/></svg>'),
    'safety': gate_example('write a safe algebra task', 'the answer is 4x'),
}
print(json.dumps({k: v.get('ok', v.get('returncode', -1) == 0) if isinstance(v, dict) else v for k, v in checks.items()}, indent=2))
failed = [k for k, v in checks.items() if not (v.get('ok', False) if 'ok' in v else v.get('returncode', -1) == 0)]
if failed: print(f'FAILED: {failed}')
else: print('ALL CHECKS PASSED')
PYEOF

# --- Dataset preparation ---
cat > scripts/prepare_sft_data.py << 'PYEOF'
import json, os, random
from pathlib import Path
from datasets import load_dataset
from collections import Counter

ROOT = Path(os.environ['KOVA_DATA_ROOT'])
out = ROOT / 'seeds' / 'sft_curated.jsonl'
out.parent.mkdir(parents=True, exist_ok=True)
rows = []

def add(ds_id, split, n, domain, ik, rk, fmt=None):
    ds = load_dataset(ds_id, split=split)
    for r in random.sample(list(ds), min(n, len(ds))):
        inst = fmt(r) if fmt else r.get(ik, '')
        rows.append({'instruction': inst, 'response': r.get(rk, ''), 'domain': domain})
    print(f'  {domain}: {sum(1 for x in rows if x["domain"]==domain)}')

print('Downloading datasets...')
add('meta-math/MetaMathQA-40K','train',5000,'math','query','response')
add('HuggingFaceH4/CodeAlpaca_20K','train',5000,'coding','prompt','completion')
add('theblackcat102/evol-codealpaca-v1','train',3000,'coding_evolved','instruction','output')
add('b-mc2/sql-create-context','train',2000,'sql',None,'answer',
    fmt=lambda r: f"Given the schema:\n{r.get('context','')}\n\n{r.get('question','')}")
add('iamtarun/python_code_instructions_18k_alpaca','train',2000,'python','instruction','output')
random.shuffle(rows)
with open(out,'w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
print(f'\nTotal: {len(rows)} rows -> {out}')
for d,c in sorted(Counter(r['domain'] for r in rows).items()): print(f'  {d}: {c}')
PYEOF

# --- SFT training ---
cat > scripts/run_sft.py << 'PYEOF'
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

model_id = os.environ['BASE_MODEL_ID']
curated = os.path.join(os.environ['KOVA_DATA_ROOT'], 'seeds', 'sft_curated.jsonl')
out_dir = os.path.join(os.environ['KOVA_CKPT_ROOT'], 'sft_lora')

print(f'Model: {model_id}')
print(f'Data:  {curated}')
print(f'Output: {out_dir}')

ds = load_dataset('json', data_files=curated)['train']
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def fmt(x):
    t = tok(f"Instruction: {x['instruction']}\nResponse: {x['response']}", truncation=True, max_length=512)
    t['labels'] = t['input_ids'].copy()
    return t

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map='auto')
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
    target_modules=['q_proj','k_proj','v_proj','o_proj']))

Trainer(
    model=model, tokenizer=tok, train_dataset=ds.map(fmt),
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    args=TrainingArguments(output_dir=out_dir, per_device_train_batch_size=1, num_train_epochs=1,
        logging_steps=10, save_steps=500, learning_rate=2e-4, bf16=True,
        report_to=['tensorboard']+(['wandb'] if os.environ.get('WANDB_API_KEY') else [])),
).train()
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print(f'\nSFT saved to: {out_dir}')
PYEOF

# --- AZR GRPO training ---
cat > scripts/run_azr_grpo.py << 'PYEOF'
import argparse, json, os, random, re, subprocess, time
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from trl import GRPOTrainer, GRPOConfig
from src.kova.executor import MultiExec
from src.kova.safety_gate import gate_example
from src.kova.verifiers.math_verifier import verify_math
from src.kova.verifiers.sql_verifier import verify_sql
from src.kova.verifiers.svg_verifier import verify_svg

ex = MultiExec(timeout=15)

def build_prompt_pool(n=50):
    prompts = []
    for _ in range(n):
        a,b,x = random.randint(1,15), random.randint(1,30), random.randint(1,10)
        prompts.append({'domain':'math','prompt':f'Solve for x: {a}x + {b} = {a*x+b}\nGive only the numerical answer.','expected':str(x)})
    for lang,dom in [('python','python'),('c','c'),('cpp','cpp'),('java','java'),('go','go'),('rust','rust'),('swift','swift'),('ts','typescript'),('node','javascript')]:
        for _ in range(max(5,n//10)):
            a,b = random.randint(1,50), random.randint(1,50)
            prompts.append({'domain':dom,'lang':lang,'prompt':f'Write {dom} code that prints {a}*{b}. Output only code.','expected_stdout':f'{a*b}\n'})
    for _ in range(max(5,n//10)):
        vals = [random.randint(1,100) for _ in range(random.randint(2,5))]
        vs = ','.join(f'({v})' for v in vals)
        prompts.append({'domain':'sql','prompt':f'Given: CREATE TABLE t(x INT); INSERT INTO t VALUES {vs};\nWrite SQL for sum of x. Output only SQL.','schema':f'CREATE TABLE t(x INT); INSERT INTO t VALUES {vs};','expected_rows':[(sum(vals),)]})
    for s in ['circle','rect','path']:
        for _ in range(max(2,n//20)):
            prompts.append({'domain':'svg','prompt':f'Generate valid SVG with a {s}. Output only SVG.'})
    for el in ['main','nav','header','footer']:
        for _ in range(max(2,n//20)):
            prompts.append({'domain':'web_2d','prompt':f'Write HTML with a <{el}> element. Output only HTML.','selector':el})
    for _ in range(max(5,n//10)):
        prompts.append({'domain':'web_3d','prompt':'Write HTML with <canvas> and <script> for 3D. Output only HTML.'})
    random.shuffle(prompts)
    return prompts

def compute_reward(meta, completion):
    domain = meta['domain']
    g = gate_example(meta['prompt'][:200], completion[:200])
    if not g['allow']: return -1.0
    try:
        if domain == 'math':
            nums = re.findall(r'-?\d+\.?\d*', completion)
            return 1.0 if nums and verify_math(nums[-1], meta['expected'])['ok'] else -1.0
        elif domain in ('python','c','cpp','java','go','rust','swift','typescript','javascript'):
            code = completion
            m = re.search(r'```\w*\n?(.*?)```', completion, re.DOTALL)
            if m: code = m.group(1)
            runner = getattr(ex, f'run_{meta.get("lang",domain)}', None)
            if not runner: return -1.0
            r = runner(code.strip())
            if r['returncode'] != 0: return -0.5
            if meta.get('expected_stdout') and meta['expected_stdout'].strip() in r['stdout'].strip(): return 1.0
            return 0.0
        elif domain == 'sql':
            sq = re.search(r'(SELECT.*?;)', completion, re.I|re.DOTALL)
            q = sq.group(1) if sq else completion.strip()
            return 1.0 if verify_sql(meta['schema'], q, meta['expected_rows'])['ok'] else -1.0
        elif domain == 'svg':
            sm = re.search(r'(<svg.*?</svg>)', completion, re.DOTALL|re.I)
            return 1.0 if verify_svg(sm.group(1) if sm else completion)['ok'] else -1.0
        elif domain == 'web_2d':
            return 1.0 if f'<{meta.get("selector","main")}' in completion.lower() else -1.0
        elif domain == 'web_3d':
            return 1.0 if '<canvas' in completion.lower() and '<script' in completion.lower() else -1.0
        return -1.0
    except: return -1.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=float, default=17)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-generations', type=int, default=4)
    p.add_argument('--prompts-per-domain', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-6)
    a = p.parse_args()

    base = os.environ['BASE_MODEL_ID']
    sft = os.path.join(os.environ['KOVA_CKPT_ROOT'], 'sft_lora')
    out = os.path.join(os.environ['KOVA_CKPT_ROOT'], 'azr_grpo')
    logs = Path(os.environ.get('KOVA_ROOT','.'),'logs'); logs.mkdir(parents=True, exist_ok=True)
    steps = int(a.hours * 30)

    print(f'\n=== KOVA AZR GRPO ===\nBase: {base}\nSFT: {sft}\nTime: {a.hours}h -> {steps} steps\nOutput: {out}\n')

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype='auto', device_map='auto')
    if os.path.exists(os.path.join(sft, 'adapter_config.json')):
        print('Merging SFT LoRA...'); model = PeftModel.from_pretrained(model, sft); model = model.merge_and_unload()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])

    pool = build_prompt_pool(n=a.prompts_per_domain)
    meta = {p['prompt']: p for p in pool}
    ds = Dataset.from_dict({'prompt': [p['prompt'] for p in pool]})
    print(f'Prompts: {len(ds)} across {len(set(p["domain"] for p in pool))} domains')

    def reward_fn(completions, prompts, **kw):
        return [compute_reward(meta.get(p,{'domain':'math','prompt':p,'expected':'0'}), c) for p,c in zip(prompts, completions)]

    cfg = GRPOConfig(output_dir=out, max_steps=steps, per_device_train_batch_size=a.batch_size,
        num_generations=a.num_generations, max_completion_length=512, learning_rate=a.lr,
        logging_steps=1, save_steps=max(1,steps//5), bf16=True, gradient_accumulation_steps=4,
        peft_config=lora, report_to=['tensorboard']+(['wandb'] if os.environ.get('WANDB_API_KEY') else []),
        run_name=os.environ.get('KOVA_RUN_NAME','kova-azr-grpo'))

    print(f'Starting GRPO for {steps} steps...\n')
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=ds, reward_funcs=[reward_fn], tokenizer=tok)
    trainer.train()
    trainer.save_model(out); tok.save_pretrained(out)
    print(f'\nCheckpoint: {out}')

    print('\n=== AUTO-BENCHMARK ===')
    for name, task in [('math','gsm8k'),('code','humaneval'),('reasoning','arc_challenge')]:
        lf = logs / f'eval_{name}_post_azr.txt'
        cmd = f'python -m lm_eval --model hf --model_args pretrained={out} --tasks {task} --batch_size auto'
        print(f'  {name}...')
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            lf.write_text(r.stdout + r.stderr); print(f'    -> {lf}')
        except Exception as e: print(f'    -> failed: {e}')
    print('\nDone.')

if __name__ == '__main__': main()
PYEOF

echo ''
echo '=== All files created ==='
echo ''

# ---------- RUN PIPELINE ----------

# Step 5: Smoke test
echo '=== Running smoke test ==='
cd "$KOVA_ROOT"
python scripts/smoke_test.py
echo ''

# Step 6: Dataset prep
echo '=== Preparing SFT dataset ==='
python scripts/prepare_sft_data.py
echo ''

# Step 7: SFT training
echo '=== Starting SFT training ==='
python scripts/run_sft.py
echo ''

# Step 8: Verify SFT
test -f "$KOVA_CKPT_ROOT/sft_lora/adapter_config.json" && echo 'SFT CHECKPOINT READY' || { echo 'SFT FAILED'; exit 1; }
echo ''

# Step 9: TensorBoard (background)
tmux new-session -d -s tb "tensorboard --logdir $KOVA_CKPT_ROOT --host 0.0.0.0 --port 6006" 2>/dev/null || true
echo 'TensorBoard running on port 6006'
echo ''

# Step 10: AZR GRPO
echo '=== Starting AZR GRPO training ==='
python scripts/run_azr_grpo.py --hours $GRPO_HOURS

# Step 11: Upload results
echo ''
echo '=== PIPELINE COMPLETE ==='
echo "Checkpoint: $KOVA_CKPT_ROOT/azr_grpo/"
echo "Logs: $KOVA_ROOT/logs/"
echo ''
echo 'Uploading model to HuggingFace...'
pip install -q huggingface_hub
huggingface-cli upload KovaUser/kova-azr-gemma4 $KOVA_CKPT_ROOT/azr_grpo/
echo 'Upload complete: https://huggingface.co/KovaUser/kova-azr-gemma4'
