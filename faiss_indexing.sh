CUDA_VISIBLE_DEVICES="3" \
python -m colbert.index_faiss \
    --index_root ./indexes/ --index_name MSMARCO.LP.L2.32x200k \
    --root ./experiments/ --experiment LPLOSS_MSMARCO