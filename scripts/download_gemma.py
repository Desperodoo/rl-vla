#!/usr/bin/env python3
"""Download Gemma pretrained weights for ACP."""
import os
os.environ["HF_TOKEN"] = open("/home/wjz/rl-vla/.hf_token_tmp").read().strip()
os.environ["http_proxy"] = "http://10.20.93.149:7890"
os.environ["https_proxy"] = "http://10.20.93.149:7890"

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Downloading Gemma model...")
m = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m",
                                          token=os.environ["HF_TOKEN"])
m.save_pretrained("checkpoints/vlaw/acp/pretrained/gemma")
print("Gemma model saved.")

print("Downloading Gemma tokenizer...")
t = AutoTokenizer.from_pretrained("google/gemma-3-270m",
                                   token=os.environ["HF_TOKEN"])
t.save_pretrained("checkpoints/vlaw/acp/pretrained/gemma")
print("Gemma tokenizer saved.")
print("ALL DONE")
