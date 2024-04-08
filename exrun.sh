torchrun --standalone --nproc_per_node 1 torchrun_alt.py \
    --model_config configs/llama_350m_el.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 16 \
    --grad_clipping 1.0 \
    --total_batch_size 64 \
    --num_training_steps 100000 \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 500 \
    --save_every 500 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer  



torchrun --standalone --nproc_per_node 1 torchrun_alt.py \
    --model_config configs/llama_900m_el.json \
    --lr 0.0005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 32 \
    --total_batch_size 512 \
    --activation_checkpointing \
    --num_training_steps 15000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 500 \
    --save_every 500 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer


torchrun --standalone --nproc_per_node 1 torchrun_alt.py \
    --model_config configs/llama_60m_el.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 100 \
    --save_every 100 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer 




    torchrun --standalone --nproc_per_node 1 torchrun_alt.py \
    --model_config configs/llama_3b_el.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 16 \
    --total_batch_size 512 \
    --activation_checkpointing \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer



torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_1b_el.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 16 \
    --total_batch_size 512 \
    --activation_checkpointing \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 500 \
    --save_every 500 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer