PRESIGNED_URL=$LLAMA_DOWNLOAD_LINK;
MODEL_PATH=$1
TARGET_FOLDER="models";

mkdir -p ${TARGET_FOLDER};

echo "Downloading LICENSE and Acceptable Usage Policy";
wget ${PRESIGNED_URL/"*"/"LICENSE"} -O ${TARGET_FOLDER}"/LICENSE";
wget ${PRESIGNED_URL/"*"/"USE_POLICY.md"} -O ${TARGET_FOLDER}"/USE_POLICY.md";

echo "Downloading tokenizer";
wget ${PRESIGNED_URL/"*"/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model";

if [[ $MODEL_PATH == *"7b"* ]]; then
    SHARD=0;
elif [[ $MODEL_PATH == *"13b"* ]]; then
    SHARD=1;
elif [[ $MODEL_PATH == *"70b"* ]]; then
    SHARD=7;
fi;

echo "Downloading $MODEL_PATH";
mkdir -p ${TARGET_FOLDER}/$MODEL_PATH;
for s in $(seq -f "0%g" 0 $SHARD);
do
    wget ${PRESIGNED_URL/"*"/"${MODEL_PATH}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}/$MODEL_PATH/consolidated.$s.pth;
done;
wget ${PRESIGNED_URL/"*"/"${MODEL_PATH}/params.json"} -O ${TARGET_FOLDER}/$MODEL_PATH/params.json;
