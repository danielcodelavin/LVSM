# GFLOP scaler for training steps
# Models with lower GFLOP counts get proportionally more training steps

# Model 1 (baseline)
model1_name = "Model-1"
model1_gflops = 43814
model1_train_steps = 350000

# Model 2
model2_name = "Model-2"
model2_gflops = 22565

# Model 3
model3_name = "Model-3"
model3_gflops = 29992

# Scale training steps inversely proportional to GFLOP count
model2_train_steps = int(model1_train_steps * (model1_gflops / model2_gflops))
model3_train_steps = int(model1_train_steps * (model1_gflops / model3_gflops))

# Print results
print(f"{'Model Name':<15} {'GFLOP Count':<15} {'Train Steps':<15}")
print("-" * 45)
print(f"{model1_name:<15} {model1_gflops:<15.1f} {model1_train_steps:<15}")
print(f"{model2_name:<15} {model2_gflops:<15.1f} {model2_train_steps:<15}")
print(f"{model3_name:<15} {model3_gflops:<15.1f} {model3_train_steps:<15}")