import os
import subprocess
from pathlib import Path

steps = [0, 100, 200, 300, 500, 700, 1000]
root_dir = './ckpts/cifar-gen-vpre'
sample_num = 50000

for step in steps:
    print(f'Generating images for step {step}')
    ckpt_path = os.path.join(root_dir, f'checkpoints/cifar-gen-{step}.pth')
    gen_path = os.path.join(root_dir, f'generates/step-{step}')
    if not os.path.exists(gen_path) or len(os.listdir(gen_path)) != sample_num:
        generate_cmd = f'python cifar-generate.py --gen-path {ckpt_path} --decoder-path dec.pth --num-samples {sample_num} --output-dir {gen_path}'
        subprocess.run(generate_cmd, shell=True)

    print(f'Evaluating images for step {step}')
    save_path = Path(root_dir) / f'results/step-{step}.txt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(save_path):
        evaluate_cmd = f'python evaluate.py --gen-path {gen_path} --cifar-path data --save-path {save_path}'
        subprocess.run(evaluate_cmd, shell=True)