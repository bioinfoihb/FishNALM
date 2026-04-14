python /nvme0/wb_ssd0/FishNALM/model_inference.py \
    -m /nvme0/wb_ssd0/Plant_DNA_LLMs-main/finetune8/dnabert-bpe-prom_300_all \
    -f /nvme0/wb_ssd0/FishNALM/test_inference.fa \
    -o predict_results.txt