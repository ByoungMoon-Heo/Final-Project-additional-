#!/bin/bash
# +
#!/bin/bash
# -

# Define parameter lists
lr=(0.0005 0.001)
alpha=(0.1 0.3 0.5 0.7 0.9)
pass_weight=(0.1 0.3 0.5 0.7 0.9)
num_attention_heads=(1 2 4)

# Nested loop to iterate through all combinations of parameters
for lr in "${lr[@]}"; do
  for alpha in "${alpha[@]}"; do
    for pass_weight in "${pass_weight[@]}"; do
      for num_attention_heads in "${num_attention_heads[@]}"; do
        # Execute the Python script and capture its output
        echo "Running with parameters: learning_rate=$lr, alpha_value=$alpha, pass_weight=$pass_weight, num_attention_head=$num_attention_heads"
        python main.py --model_type WAVERec --data_name Beauty --lr "$lr" --alpha "$alpha" --pass_weight "$pass_weight" --num_attention_heads "$num_attention_heads" --train_name WAVERec_Beauty
      done
    done
  done
done
