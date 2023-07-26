PRESIGNED_URL=$LLAMA_DOWNLOAD_LINK;
MODEL_PATH=$1
TARGET_FOLDER="models"

mkdir -p ${TARGET_FOLDER};

echo "Downloading LICENSE and Acceptable Usage Policy";
wget ${PRESIGNED_URL/"*"/"LICENSE"} -O ${TARGET_FOLDER}"/LICENSE";
wget ${PRESIGNED_URL/"*"/"USE_POLICY.md"} -O ${TARGET_FOLDER}"/USE_POLICY.md";

if [[ $MODEL_PATH == *"7b"* ]]; then
    SHARD=0;
    SHORT="7B";
elif [[ $MODEL_PATH == *"13b"* ]]; then
    SHARD=1;
    SHORT="13B";
elif [[ $MODEL_PATH == *"70b"* ]]; then
    SHARD=7;
    SHORT="70B";
fi;

echo "Downloading $MODEL_PATH";
mkdir -p ${TARGET_FOLDER}/$MODEL_PATH;

echo "Downloading tokenizer";
wget ${PRESIGNED_URL/"*"/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model";
for s in $(seq -f "0%g" 0 $SHARD);
do
    wget ${PRESIGNED_URL/"*"/"${MODEL_PATH}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}/$MODEL_PATH/consolidated.$s.pth;
done;
wget ${PRESIGNED_URL/"*"/"${MODEL_PATH}/params.json"} -O ${TARGET_FOLDER}/$MODEL_PATH/params.json;
mv ${TARGET_FOLDER}/$MODEL_PATH ${x}/$SHORT;
python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir models --model_size 7B --output_dir models/llama-2-7b-hf;
