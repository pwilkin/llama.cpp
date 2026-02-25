#!/bin/bash

PYTHONPATH=/devel/tools/llama.cpp/gguf-py /home/ilintar/venv/bin/python -c "
import gguf
reader = gguf.GGUFReader('LFM2.5-1.2B-Instruct-IQ3_KL.gguf', 'r')
field = reader.fields.get('iq3_kl.codebooks')
if field:
    data = field.parts[-1]
    print(f'Codebook data length: {len(data)}')
    print(f'First 100 bytes: {list(data[:100])}')
    print(f'Non-zero count: {sum(1 for x in data if x != 0)}')
else:
    print('No iq3_kl.codebooks field found!')
"
