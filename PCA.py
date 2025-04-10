import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

with open("train_embeddings.pkl", "rb") as f:
    res_loaded = pickle.load(f)

print(res_loaded)
train_array = np.vstack(res_loaded.array)
print(train_array.shape)

with open("test_embeddings.pkl", "rb") as f:
    res_loaded = pickle.load(f)

print(res_loaded)
test_array = np.vstack(res_loaded.array)
print(test_array.shape)

# Step 1: PCA with fixed number of components
n_components = 100
pca = PCA(n_components=n_components)
pca.fit(train_array)

# Step 2: Transform both train and test sets
train_pca = pca.transform(train_array)
test_pca = pca.transform(test_array)
np.save('LLM_PCA/train_pca.npy', train_pca)
np.save('LLM_PCA/test_pca.npy', test_pca)

# Step 3: Plot cumulative explained variance (for all components)
pca_full = PCA().fit(train_array)  # Fit again to get all components' explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.axvline(x=n_components - 1, color='g', linestyle='--', label=f'{n_components} Components')
plt.title("Cumulative Explained Variance from PCA")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LLM_PCA/Cumulative_Explained_Var.png")
plt.show()

# Step 3: Plot the singular values
plt.figure(figsize=(8, 5))
plt.plot(pca.singular_values_, marker='o')
plt.title("Singular Values from PCA on train_array")
plt.xlabel("Component Index")
plt.ylabel("Singular Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("LLM_PCA/singular_vals.png")
plt.show()
