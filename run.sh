#!/bin/bash

python pipeline.py \
  --dataset datasets/cleaned/DrHCM_cleaned.csv \
  --output results/DrHCM_all_prompt_types.csv \
  --prompt-models llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b \
  --answer-models llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b \
  --prompt-types "chain of thought" "trigger chain of thought" "self consistency" "prompt chaining" "react" "tree of thoughts" "role based" "metacognitive prompting" "uncertainty based prompting" "guided prompting" \
  --prompts-per-type 10 \
  --num-questions 100 



python pipeline.py \
  --dataset datasets/cleaned/DrHCM_cleaned.csv \
  --output results/DrHCM_all_prompt_types.csv \
  --prompt-models llama-3.2-1b \
  --answer-models llama-3.2-1b \
  --prompt-types "chain of thought" "trigger chain of thought" "self consistency" "prompt chaining" "react" "tree of thoughts" "role based" "metacognitive prompting" "uncertainty based prompting" "guided prompting" \
  --prompts-per-type 10 \
  --num-questions 100 


python pipeline.py \
  --dataset datasets/cleaned/DrHCM_cleaned.csv \
  --output results/DrHCM_all_prompt_types.csv \
  --prompt-models llama-3.2-1b \
  --answer-models llama-3.2-1b \
  --prompt-types  "uncertainty based prompting" "guided prompting" \
  --prompts-per-type 2 \
  --num-questions 5 