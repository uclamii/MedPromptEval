#!/bin/bash
#Evaluate different prompt types across same model 
python pipeline.py \  --dataset datasets/cleaned/DrHCM_cleaned.csv \                                                           
  --output results/DrHCM_all_prompt_types_llama_3.2.csv \
  --prompt-models llama-3.2-1b \
  --answer-models llama-3.2-1b \
  --prompt-types "chain of thought" "trigger chain of thought" "self consistency" "prompt chaining" "react" "tree of thoughts" "role based" "metacognitive prompting" "uncertainty based prompting" "guided prompting" \
  --prompts-per-type 10 \
  --num-questions 100 


#Evaluate different models across same prompt types 
python pipeline.py \  --dataset datasets/cleaned/medquad_cleaned.csv \                                                           
  --output results/medquad_self_consistency_5_models.csv \
  --prompt-models llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b \
  --answer-models llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b \
  --prompt-types "self consistency" \
  --prompts-per-type 10 \
  --num-questions 100 

#Comprehensive evaluation of prompt types and models 
python pipeline.py \  --dataset datasets/cleaned/pubmedqa_cleaned.csv \                                                           
  --output results/pubmedqa_5_prompt_types_5_models.csv \
  --prompt-models llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b \
  --answer-models llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b \
  --prompt-types "chain of thought" "react" "tree of thoughts" "role based" "metacognitive prompting" \
  --prompts-per-type 10 \
  --num-questions 100 