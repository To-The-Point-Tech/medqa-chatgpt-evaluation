wget https://nlp.stanford.edu/projects/myasu/LinkBERT/data.zip
unzip data.zip
rm data.zip

ls data/seqcls/hoc_hf data/hoc
mv data/mc/medqa_usmle_hf medqa 
mv data/seqcls/bioasq_hf bioasq 
mv data/seqcls/pubmedqa_hf pubmed
mv data/mc/mmlu_hf/professional_medicine data/mmlu

rm -rf data/mc
rm -rf data/seqcls
rm -rf data/tokcls
rm -rf data/qa

python utils/download_medmcqa.py