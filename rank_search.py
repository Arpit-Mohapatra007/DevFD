from train import train_continual

results = {}
for r in [32, 64, 128]:
    print(f"\n######## rank {r} ########")
    M = train_continual(rank=r, epochs=20, n_tasks=4, holdout=True)
    last = M[3]
    aa = last[:4].mean()
    af = sum(M[i, i] - last[i] for i in range(3)) / 3
    results[r] = (aa, af)

print("\nrank   AA     AF")
for r, (aa, af) in results.items():
    print(f"{r:>4}  {aa:6.2f} {af:6.2f}")
