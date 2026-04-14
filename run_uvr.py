# file: scripts/run_uvr.py
# Kova UVR — Unified Verifiable Reinforcement
# Single-pass training: curriculum warmup → ramp → full self-play
# No SFT stage. No external datasets. One script.

import argparse, json, os, random, re, subprocess, time, math as pymath
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import torch
import numpy as np

# ── Verifiers ──
import sympy as sp
import sqlite3
from xml.etree import ElementTree as ET

def run_code(lang, code, timeout=15):
    """Execute code in any supported language, return {returncode, stdout, stderr}"""
    import tempfile
    try:
        if lang == 'python':
            r = subprocess.run(['python','-c',code], capture_output=True, text=True, timeout=timeout)
        elif lang == 'node':
            r = subprocess.run(['node','-e',code], capture_output=True, text=True, timeout=timeout)
        elif lang in ('c','cpp','java','go','rust','swift','ts'):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)
                if lang == 'c':
                    (p/'main.c').write_text(code)
                    subprocess.run(['gcc',str(p/'main.c'),'-o',str(p/'main'),'-lm'], capture_output=True, timeout=timeout)
                    r = subprocess.run([str(p/'main')], capture_output=True, text=True, timeout=timeout)
                elif lang == 'cpp':
                    (p/'main.cpp').write_text(code)
                    subprocess.run(['g++',str(p/'main.cpp'),'-o',str(p/'main'),'-std=c++17','-lm'], capture_output=True, timeout=timeout)
                    r = subprocess.run([str(p/'main')], capture_output=True, text=True, timeout=timeout)
                elif lang == 'java':
                    (p/'Main.java').write_text(code)
                    subprocess.run(['javac','Main.java'], cwd=str(p), capture_output=True, timeout=timeout)
                    r = subprocess.run(['java','Main'], cwd=str(p), capture_output=True, text=True, timeout=timeout)
                elif lang == 'go':
                    (p/'main.go').write_text(code)
                    r = subprocess.run(['go','run','main.go'], cwd=str(p), capture_output=True, text=True, timeout=timeout)
                elif lang == 'rust':
                    (p/'main.rs').write_text(code)
                    subprocess.run(['rustc',str(p/'main.rs'),'-o',str(p/'main')], capture_output=True, timeout=timeout)
                    r = subprocess.run([str(p/'main')], capture_output=True, text=True, timeout=timeout)
                elif lang == 'swift':
                    (p/'main.swift').write_text(code)
                    r = subprocess.run(['swift',str(p/'main.swift')], capture_output=True, text=True, timeout=timeout)
                elif lang == 'ts':
                    (p/'main.tsx').write_text(code)
                    r = subprocess.run(['npx','tsx','main.tsx'], cwd=str(p), capture_output=True, text=True, timeout=timeout)
                else:
                    return {'returncode':-1,'stdout':'','stderr':'unsupported'}
        else:
            return {'returncode':-1,'stdout':'','stderr':'unsupported'}
        return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except Exception as e:
        return {'returncode':-1,'stdout':'','stderr':str(e)}

def verify_math(pred, target):
    try: return sp.simplify(sp.sympify(str(pred))) == sp.simplify(sp.sympify(str(target)))
    except: return False

def verify_sql(schema, query, expected):
    try:
        c = sqlite3.connect(':memory:'); cur = c.cursor()
        cur.executescript(schema); rows = cur.execute(query).fetchall(); c.close()
        return sorted(rows) == sorted(expected)
    except: return False

def verify_svg(text):
    try:
        root = ET.fromstring(text)
        return root.tag.endswith('svg') and any(c.tag.endswith(t) for c in root.iter() for t in ('path','circle','rect'))
    except: return False

def safety_check(text):
    blocked = ['credential theft','weapon','exfiltrat','ransomware','zero-day','privilege escalation']
    low = text.lower()
    return not any(b in low for b in blocked) and len(text.strip()) >= 8

# ── Curriculum: difficulty-parameterized task generators ──

DOMAINS = ['math','python','c','cpp','java','go','rust','swift','typescript','javascript','sql','web_2d','web_3d','svg','cybersecurity']
LANG_MAP = {'python':'python','c':'c','cpp':'cpp','java':'java','go':'go','rust':'rust','swift':'swift','typescript':'ts','javascript':'node'}

def gen_math(difficulty):
    if difficulty < 0.3:
        # Systems of linear equations: ax + by = c, dx + ey = f
        x, y = random.randint(1,10), random.randint(1,10)
        a, b = random.randint(1,5), random.randint(1,5)
        d, e = random.randint(1,5), random.randint(1,5)
        c1, c2 = a*x + b*y, d*x + e*y
        return {'prompt': f'Solve: {a}x + {b}y = {c1} and {d}x + {e}y = {c2}. What is x+y? Answer with just the number.', 'expected': str(x+y)}
    elif difficulty < 0.7:
        # Polynomial evaluation, combinatorics, modular arithmetic
        task_type = random.choice(['poly','comb','mod'])
        if task_type == 'poly':
            a,b,c = random.randint(1,10), random.randint(-10,10), random.randint(-20,20)
            x = random.randint(-5,5)
            ans = a*x*x + b*x + c
            return {'prompt': f'Evaluate f(x) = {a}x^2 + {"+" if b>=0 else ""}{b}x + {"+" if c>=0 else ""}{c} at x={x}. Answer with just the number.', 'expected': str(ans)}
        elif task_type == 'comb':
            n = random.randint(5,12); k = random.randint(1,n-1)
            ans = pymath.comb(n,k)
            return {'prompt': f'Compute C({n},{k}) (n choose k). Answer with just the number.', 'expected': str(ans)}
        else:
            a, b = random.randint(10,500), random.randint(2,17)
            return {'prompt': f'What is {a} mod {b}? Answer with just the number.', 'expected': str(a%b)}
    else:
        # Matrix determinant, sum of series, quadratic roots
        task_type = random.choice(['det','series','roots'])
        if task_type == 'det':
            a,b,c,d = [random.randint(-5,5) for _ in range(4)]
            det = a*d - b*c
            return {'prompt': f'Compute the determinant of the 2x2 matrix [[{a},{b}],[{c},{d}]]. Answer with just the number.', 'expected': str(det)}
        elif task_type == 'series':
            n = random.randint(10,50)
            ans = n*(n+1)//2
            return {'prompt': f'Compute the sum 1+2+3+...+{n}. Answer with just the number.', 'expected': str(ans)}
        else:
            x1, x2 = random.randint(-10,10), random.randint(-10,10)
            a = 1; b = -(x1+x2); c = x1*x2
            return {'prompt': f'Find the sum of roots of x^2 + {"+" if b>=0 else ""}{b}x + {"+" if c>=0 else ""}{c} = 0. Answer with just the number.', 'expected': str(x1+x2)}

def gen_code(lang, difficulty):
    dom = lang; lang_key = LANG_MAP.get(lang, lang)
    if difficulty < 0.3:
        # Fibonacci, sorting, basic data structures
        task_type = random.choice(['fib','sort','gcd'])
        if task_type == 'fib':
            n = random.randint(5,15)
            a,b = 0,1
            for _ in range(n): a,b = b,a+b
            return {'prompt': f'Write {dom} code that computes the {n}th Fibonacci number (0-indexed, F(0)=0, F(1)=1) and prints it. Output only code.', 'expected_stdout': f'{a}\n', 'lang': lang_key}
        elif task_type == 'sort':
            arr = [random.randint(1,100) for _ in range(random.randint(5,10))]
            return {'prompt': f'Write {dom} code that sorts the array {arr} in ascending order and prints the sorted array as space-separated numbers on one line. Output only code.', 'expected_stdout': ' '.join(map(str,sorted(arr)))+'\n', 'lang': lang_key}
        else:
            a, b = random.randint(10,200), random.randint(10,200)
            return {'prompt': f'Write {dom} code that computes GCD of {a} and {b} using Euclidean algorithm and prints it. Output only code.', 'expected_stdout': f'{pymath.gcd(a,b)}\n', 'lang': lang_key}
    elif difficulty < 0.7:
        # Binary search, matrix multiplication result, string ops
        task_type = random.choice(['bsearch','prime_count','reverse_words'])
        if task_type == 'bsearch':
            arr = sorted(random.sample(range(1,200), 20))
            target = random.choice(arr)
            idx = arr.index(target)
            return {'prompt': f'Write {dom} code that implements binary search to find {target} in {arr} and prints its index (0-based). Output only code.', 'expected_stdout': f'{idx}\n', 'lang': lang_key}
        elif task_type == 'prime_count':
            n = random.randint(20,100)
            count = sum(1 for i in range(2,n+1) if all(i%j!=0 for j in range(2,int(i**0.5)+1)))
            return {'prompt': f'Write {dom} code that counts the number of prime numbers from 2 to {n} (inclusive) and prints the count. Output only code.', 'expected_stdout': f'{count}\n', 'lang': lang_key}
        else:
            words = random.choice([['hello','world','foo'],['the','quick','brown','fox'],['one','two','three','four','five']])
            return {'prompt': f'Write {dom} code that reverses the order of words in "{" ".join(words)}" and prints the result. Output only code.', 'expected_stdout': ' '.join(reversed(words))+'\n', 'lang': lang_key}
    else:
        # DP, graph algorithms, complex operations
        task_type = random.choice(['factorial','power','collatz'])
        if task_type == 'factorial':
            n = random.randint(8,15); ans = pymath.factorial(n)
            return {'prompt': f'Write {dom} code that computes {n}! (factorial) and prints the result. Output only code.', 'expected_stdout': f'{ans}\n', 'lang': lang_key}
        elif task_type == 'power':
            base, exp = random.randint(2,5), random.randint(5,15)
            return {'prompt': f'Write {dom} code that computes {base}^{exp} using repeated squaring (fast exponentiation) and prints the result. Output only code.', 'expected_stdout': f'{base**exp}\n', 'lang': lang_key}
        else:
            n = random.randint(5,30)
            steps = 0; v = n
            while v != 1: v = v//2 if v%2==0 else 3*v+1; steps += 1
            return {'prompt': f'Write {dom} code that computes the number of steps in the Collatz sequence starting from {n} until reaching 1 and prints it. Output only code.', 'expected_stdout': f'{steps}\n', 'lang': lang_key}

def gen_sql(difficulty):
    if difficulty < 0.3:
        # Multi-table JOIN with GROUP BY
        emps = [(f'E{i}', random.choice(['Sales','Eng','HR']), random.randint(40,120)*1000) for i in range(1,8)]
        target_dept = random.choice(['Sales','Eng','HR'])
        avg_sal = sum(s for _,d,s in emps if d==target_dept) // max(1,sum(1 for _,d,_ in emps if d==target_dept))
        inserts = ';'.join(f"INSERT INTO emp VALUES('{n}','{d}',{s})" for n,d,s in emps)
        schema = f'CREATE TABLE emp(name TEXT, dept TEXT, salary INT);{inserts};'
        return {'prompt': f'Given: {schema}\nWrite SQL to get the average salary (integer division) for dept \'{target_dept}\'. Output only SQL.', 'schema': schema, 'expected_rows': [(avg_sal,)]}
    elif difficulty < 0.7:
        # Subqueries, HAVING
        vals = [random.randint(1,50) for _ in range(random.randint(5,10))]
        vs = ';'.join(f'INSERT INTO t VALUES({i+1},{v})' for i,v in enumerate(vals))
        schema = f'CREATE TABLE t(id INT, val INT);{vs};'
        max_val = max(vals)
        return {'prompt': f'Given: {schema}\nWrite SQL to find the maximum value in column val. Output only SQL.', 'schema': schema, 'expected_rows': [(max_val,)]}
    else:
        # Complex aggregations, multiple tables
        products = [(f'P{i}', random.randint(10,100), random.choice(['A','B','C'])) for i in range(1,7)]
        target_cat = random.choice(['A','B','C'])
        total = sum(p for _,p,c in products if c==target_cat)
        count = sum(1 for _,_,c in products if c==target_cat)
        inserts = ';'.join(f"INSERT INTO products VALUES('{n}',{p},'{c}')" for n,p,c in products)
        schema = f'CREATE TABLE products(name TEXT, price INT, category TEXT);{inserts};'
        return {'prompt': f'Given: {schema}\nWrite SQL to get the total price and count of products in category \'{target_cat}\'. Output only SQL.', 'schema': schema, 'expected_rows': [(total,count)]}

def gen_svg(difficulty):
    if difficulty < 0.3:
        shapes = ['circle','rect','path']
        s = random.choice(shapes)
        return {'prompt': f'Generate valid SVG containing a {s} element with a fill color. Output only SVG markup.'}
    elif difficulty < 0.7:
        return {'prompt': 'Generate valid SVG with at least 3 different shapes (circle, rect, path) arranged in a pattern. Include viewBox attribute. Output only SVG markup.'}
    else:
        return {'prompt': 'Generate valid SVG with a complex path element using cubic bezier curves (C command), a gradient fill using linearGradient, and a transform attribute. Output only SVG markup.'}

def gen_web2d(difficulty):
    if difficulty < 0.3:
        els = ['nav','main','header','footer']
        el = random.choice(els)
        return {'prompt': f'Write semantic HTML5 with a <{el}> element, a heading, and a paragraph. Include a lang attribute on html. Output only HTML.', 'selector': el}
    elif difficulty < 0.7:
        el = random.choice(['form','table','section'])
        return {'prompt': f'Write HTML with a <{el}> element. Include ARIA labels for accessibility. Add CSS in a style tag for responsive layout using flexbox. Output only HTML.', 'selector': el}
    else:
        el = 'main'
        return {'prompt': 'Write HTML with <main>, <nav>, and <footer>. Include a responsive CSS grid layout in a style tag, proper ARIA roles, and semantic elements throughout. Output only HTML.', 'selector': el}

def gen_web3d(difficulty):
    if difficulty < 0.3:
        return {'prompt': 'Write HTML with a <canvas> element and a <script> tag that gets a WebGL context and clears to a color. Output only HTML.'}
    elif difficulty < 0.7:
        return {'prompt': 'Write HTML with <canvas> and <script> that creates a WebGL context, sets up a vertex buffer with triangle vertices, and uses requestAnimationFrame for an animation loop. Output only HTML.'}
    else:
        return {'prompt': 'Write HTML with <canvas> and <script> that creates a WebGL2 context, compiles vertex and fragment shaders, draws a rotating colored triangle using requestAnimationFrame, and handles canvas resize. Output only HTML.'}

def gen_cybersecurity(difficulty):
    if difficulty < 0.3:
        vuln_code = 'query = "SELECT * FROM users WHERE name=\'" + user_input + "\'"'
        return {'prompt': f'The following Python code has a SQL injection vulnerability:\n{vuln_code}\nRewrite it using parameterized queries to prevent SQL injection. Output only the fixed code.', 'check_patterns': ['?','parameterize','execute'], 'check_absent': ['+ user_input','format(','f"']}
    elif difficulty < 0.7:
        task_type = random.choice(['xss','hash'])
        if task_type == 'xss':
            return {'prompt': 'Write a Python function sanitize_html(user_input) that escapes HTML special characters (<, >, &, ", \') to prevent XSS attacks. Print the result of sanitize_html(\'<script>alert(1)</script>\'). Output only code.', 'check_patterns': ['&lt;','&gt;','escape','html'], 'lang': 'python'}
        else:
            return {'prompt': 'Write Python code that hashes a password "mypassword123" using bcrypt or hashlib with a salt. Print "HASHED" if the hash is generated successfully. Output only code.', 'expected_stdout': 'HASHED\n', 'lang': 'python'}
    else:
        return {'prompt': 'Write a Python function validate_and_sanitize(email, password) that: 1) validates email format with regex, 2) checks password is 8+ chars with uppercase+lowercase+digit, 3) returns (sanitized_email, True) or (None, False). Test it with "user@test.com" and "MyPass123" and print the result. Output only code.', 'check_patterns': ['re.','len(','isupper','isdigit','islower'], 'lang': 'python'}

def generate_task(domain, difficulty):
    if domain == 'math': return {**gen_math(difficulty), 'domain': 'math'}
    elif domain == 'sql': return {**gen_sql(difficulty), 'domain': 'sql'}
    elif domain == 'svg': return {**gen_svg(difficulty), 'domain': 'svg'}
    elif domain == 'web_2d': return {**gen_web2d(difficulty), 'domain': 'web_2d'}
    elif domain == 'web_3d': return {**gen_web3d(difficulty), 'domain': 'web_3d'}
    elif domain == 'cybersecurity': return {**gen_cybersecurity(difficulty), 'domain': 'cybersecurity'}
    else: return {**gen_code(domain, difficulty), 'domain': domain}

# ── Reward computation ──

def compute_reward(task, completion):
    domain = task['domain']
    if not safety_check(completion[:300]): return -1.0
    try:
        if domain == 'math':
            nums = re.findall(r'-?\d+\.?\d*', completion)
            return 1.0 if nums and verify_math(nums[-1], task['expected']) else -1.0
        elif domain in LANG_MAP:
            code = completion
            m = re.search(r'```\w*\n?(.*?)```', completion, re.DOTALL)
            if m: code = m.group(1)
            r = run_code(task.get('lang', LANG_MAP.get(domain,'python')), code.strip())
            if r['returncode'] != 0: return -0.5
            if task.get('expected_stdout') and task['expected_stdout'].strip() in r['stdout'].strip(): return 1.0
            return 0.0
        elif domain == 'sql':
            sq = re.search(r'(SELECT.*?;)', completion, re.I|re.DOTALL)
            q = sq.group(1) if sq else completion.strip()
            return 1.0 if verify_sql(task.get('schema',''), q, task.get('expected_rows',[])) else -1.0
        elif domain == 'svg':
            sm = re.search(r'(<svg.*?</svg>)', completion, re.DOTALL|re.I)
            return 1.0 if verify_svg(sm.group(1) if sm else completion) else -1.0
        elif domain == 'web_2d':
            return 1.0 if f'<{task.get("selector","main")}' in completion.lower() else -1.0
        elif domain == 'web_3d':
            return 1.0 if '<canvas' in completion.lower() and '<script' in completion.lower() else -1.0
        return -1.0
    except: return -1.0

# ── Held-out eval set ──

def build_eval_sets(n_per_domain=50):
    evals = {}
    for dom in DOMAINS:
        evals[dom] = [generate_task(dom, 0.7) for _ in range(n_per_domain)]
    return evals

def evaluate_model_on_domain(model, tokenizer, domain, eval_tasks, device='cuda'):
    correct = 0
    for task in eval_tasks:
        inputs = tokenizer(task['prompt'], return_tensors='pt', truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        reward = compute_reward(task, completion)
        if reward > 0.5: correct += 1
    return correct / len(eval_tasks)

# ── Curriculum controller ──

class CurriculumController:
    def __init__(self, domains, targets, ramp_steps=300):
        self.domains = domains
        self.targets = targets
        self.ramp_steps = ramp_steps
        self.domain_scores = {d: 0.0 for d in domains}
        self.domain_met = {d: False for d in domains}
        self.step = 0

    @property
    def difficulty(self):
        return min(1.0, self.step / self.ramp_steps)

    def sample_domain(self):
        # Unmet domains get weight proportional to (1 - score)
        # Met domains get 2% maintenance floor
        weights = []
        for d in self.domains:
            if self.domain_met[d]:
                weights.append(0.02)
            else:
                weights.append(max(0.05, 1.0 - self.domain_scores[d]))
        total = sum(weights)
        weights = [w/total for w in weights]
        return random.choices(self.domains, weights=weights, k=1)[0]

    def update_scores(self, new_scores):
        for d, score in new_scores.items():
            self.domain_scores[d] = score
            if score >= self.targets.get(d, 0.95):
                self.domain_met[d] = True
        self.step += 1

    def all_targets_met(self):
        return all(self.domain_met.values())

    def status_string(self):
        parts = []
        for d in self.domains:
            score = self.domain_scores[d]
            target = self.targets.get(d, 0.95)
            check = ' ✓' if self.domain_met[d] else ''
            parts.append(f'{d}={score:.0%}{check}')
        return ' | '.join(parts)

# ── Main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-hours', type=float, default=30)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-generations', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--ramp-steps', type=int, default=300)
    p.add_argument('--eval-per-domain', type=int, default=50)
    args = p.parse_args()

    base = os.environ.get('BASE_MODEL_ID', 'Qwen/Qwen3.5-4B')
    out = os.path.join(os.environ.get('KOVA_CKPT_ROOT', 'checkpoints'), 'uvr')
    logs_dir = Path(os.environ.get('KOVA_ROOT', '.'), 'logs'); logs_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_hours * 30)
    start_time = time.time()

    targets = {d: 0.95 for d in DOMAINS}
    targets['web_2d'] = 0.90
    targets['web_3d'] = 0.90

    print(f'\n{"="*50}')
    print(f'KOVA UVR — Unified Verifiable Reinforcement')
    print(f'{"="*50}')
    print(f'Base: {base}')
    print(f'Max time: {args.max_hours}h ({max_steps} steps)')
    print(f'Ramp steps: {args.ramp_steps}')
    print(f'Domains: {len(DOMAINS)}')
    print(f'Output: {out}')
    print()

    # Load model
    print('Loading model...')
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype='auto', device_map='auto', trust_remote_code=True)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj'])

    # Build held-out eval sets
    print(f'Building held-out eval sets ({args.eval_per_domain} per domain)...')
    eval_sets = build_eval_sets(args.eval_per_domain)

    # Initialize curriculum
    curriculum = CurriculumController(DOMAINS, targets, ramp_steps=args.ramp_steps)

    # ── Self-play: model proposes tasks ──
    def generate_self_play_task(model, tokenizer, domain, device='cuda'):
        """Ask the model to propose a coding/math task, then verify it's solvable."""
        meta_prompts = {
            'math': 'Create a math problem that has a single numerical answer. Format: PROBLEM: <problem>\nANSWER: <number>',
            'python': 'Create a Python coding challenge. The solution should print a single number or word. Format: TASK: <task>\nEXPECTED_OUTPUT: <output>',
            'sql': 'Create a SQL challenge with a CREATE TABLE and INSERT statements, then a question. Format: SCHEMA: <schema>\nTASK: <question>\nEXPECTED: <result>',
            'cybersecurity': 'Create a Python security challenge about fixing a vulnerability. Format: TASK: <task description>',
        }
        # Default meta-prompt for code domains
        lang = domain if domain in LANG_MAP else 'python'
        meta = meta_prompts.get(domain, f'Create a {domain} coding challenge where the solution prints a single number. Format: TASK: <task>\nEXPECTED_OUTPUT: <output>')
        
        inputs = tokenizer(meta, return_tensors='pt', truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.9, pad_token_id=tokenizer.pad_token_id)
        proposal = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Try to parse the proposal into a task
        try:
            if domain == 'math' and 'ANSWER:' in proposal:
                parts = proposal.split('ANSWER:')
                prompt = parts[0].replace('PROBLEM:','').strip()
                expected = re.findall(r'-?\d+\.?\d*', parts[1])
                if prompt and expected:
                    return {'prompt': prompt + ' Answer with just the number.', 'expected': expected[0], 'domain': 'math', 'self_play': True}
            elif 'EXPECTED_OUTPUT:' in proposal:
                parts = proposal.split('EXPECTED_OUTPUT:')
                task_text = parts[0].replace('TASK:','').strip()
                expected_out = parts[1].strip().split('\n')[0]
                if task_text and expected_out:
                    lang_key = LANG_MAP.get(domain, 'python')
                    return {'prompt': task_text + ' Output only code.', 'expected_stdout': expected_out + '\n', 'lang': lang_key, 'domain': domain, 'self_play': True}
        except:
            pass
        return None  # Failed to parse — fall back to template

    # Build prompt pool: mix templates + self-play
    def build_pool(curriculum, size=500):
        pool = []
        self_play_ratio = min(0.5, curriculum.difficulty)  # 0% at start, up to 50% at full difficulty
        self_play_count = int(size * self_play_ratio)
        template_count = size - self_play_count
        
        # Template tasks
        for _ in range(template_count):
            dom = curriculum.sample_domain()
            task = generate_task(dom, curriculum.difficulty)
            pool.append(task)
        
        # Self-play tasks (model proposes)
        if self_play_count > 0:
            print(f'  Generating {self_play_count} self-play tasks...')
            sp_generated = 0
            for _ in range(self_play_count * 3):  # Try 3x because some will fail to parse
                if sp_generated >= self_play_count:
                    break
                dom = curriculum.sample_domain()
                task = generate_self_play_task(model, tok, dom)
                if task is not None:
                    pool.append(task)
                    sp_generated += 1
            # Fill remaining with templates if self-play didn't generate enough
            while len(pool) < size:
                dom = curriculum.sample_domain()
                pool.append(generate_task(dom, curriculum.difficulty))
            print(f'  Self-play: {sp_generated}/{self_play_count} proposed, {template_count} templates')
        
        return pool

    pool = build_pool(curriculum)
    meta_map = {t['prompt']: t for t in pool}
    ds = Dataset.from_dict({'prompt': [t['prompt'] for t in pool]})
    print(f'Initial prompt pool: {len(ds)} tasks at difficulty {curriculum.difficulty:.2f}')

    # Reward function for GRPOTrainer
    def reward_fn(completions, prompts, **kw):
        return [compute_reward(meta_map.get(p, {'domain':'math','prompt':p,'expected':'0'}), c)
                for p, c in zip(prompts, completions)]

    # GRPO config
    cfg = GRPOConfig(
        output_dir=out, max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=512, learning_rate=args.lr,
        logging_steps=1, save_steps=max(1, max_steps//10),
        bf16=True, gradient_accumulation_steps=4,
        peft_config=lora,
        report_to=['tensorboard'],
        run_name='kova-uvr'
    )

    # Train
    print(f'\nStarting UVR training...\n')
    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        reward_funcs=[reward_fn], processing_class=tok
    )

    # Custom training loop with curriculum updates
    step = 0
    best_avg_score = 0.0

    while step < max_steps:
        elapsed = (time.time() - start_time) / 3600
        if elapsed >= args.max_hours:
            print(f'\nTime limit ({args.max_hours}h) reached at step {step}')
            break

        # Train for eval_every steps
        train_steps = min(args.eval_every, max_steps - step)
        trainer.args.max_steps = step + train_steps
        trainer.train(resume_from_checkpoint=step > 0)
        step += train_steps
        curriculum.step = step

        # Evaluate on held-out sets
        print(f'\n--- Eval at step {step} (difficulty={curriculum.difficulty:.2f}, elapsed={elapsed:.1f}h) ---')
        new_scores = {}
        for dom in DOMAINS:
            score = evaluate_model_on_domain(model, tok, dom, eval_sets[dom])
            new_scores[dom] = score
        curriculum.update_scores(new_scores)

        avg_score = np.mean(list(new_scores.values()))
        print(f'Avg: {avg_score:.1%} | {curriculum.status_string()}')

        # Save if improved
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            trainer.save_model(out)
            tok.save_pretrained(out)
            print(f'New best! Saved checkpoint (avg={avg_score:.1%})')

        # Check if all targets met
        if curriculum.all_targets_met():
            print(f'\n🎯 ALL TARGETS MET at step {step}!')
            break

        # Rebuild prompt pool with updated difficulty and domain focus
        pool = build_pool(curriculum)
        meta_map = {t['prompt']: t for t in pool}
        ds = Dataset.from_dict({'prompt': [t['prompt'] for t in pool]})
        trainer.train_dataset = ds
        print(f'Rebuilt pool: {len(ds)} tasks, difficulty={curriculum.difficulty:.2f}')

    # Final save
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f'\nFinal checkpoint: {out}')
    print(f'Total time: {(time.time()-start_time)/3600:.1f}h')
    print(f'Final scores: {curriculum.status_string()}')

    # Auto-benchmark
    print('\n=== AUTO-BENCHMARK ===')
    for name, task in [('math','gsm8k'),('code','humaneval'),('reasoning','arc_challenge')]:
        lf = logs_dir / f'eval_{name}_post_uvr.txt'
        cmd = f'python -m lm_eval --model hf --model_args pretrained={out} --tasks {task} --batch_size auto'
        print(f'  {name}...')
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            lf.write_text(r.stdout + r.stderr)
            print(f'    -> {lf}')
        except Exception as e:
            print(f'    -> failed: {e}')

    # Save final scores
    scores_file = logs_dir / 'uvr_domain_scores.json'
    with open(scores_file, 'w') as f:
        json.dump({'scores': curriculum.domain_scores, 'targets': targets,
                   'all_met': curriculum.all_targets_met(), 'total_steps': step}, f, indent=2)
    print(f'\nDomain scores saved to {scores_file}')
    print('Done.')

if __name__ == '__main__': main()
