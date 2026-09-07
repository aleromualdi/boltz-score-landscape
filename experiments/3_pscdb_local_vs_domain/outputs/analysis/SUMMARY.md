# Exp 3 — Full PSCDB local vs domain

OK systems: **219**

## By class (ΔAUROC = Boltz − ANM)

| class | n | med AUROC B/A | med ΔAUROC [95% CI] | med ΔAUPRC | med Δtop-m | wins B/A | lock |
|---|---:|---:|---:|---:|---:|---:|:---:|
| IL | 78 | 0.905/0.818 | +0.033 [+0.015, +0.074] | +0.015 | +0.000 | 54/24 | yes |
| CL | 68 | 0.894/0.704 | +0.133 [+0.092, +0.184] | +0.071 | +0.000 | 56/12 | yes |
| CD | 31 | 0.539/0.571 | -0.049 [-0.137, -0.016] | -0.049 | -0.055 | 10/21 | no |
| ID | 42 | 0.624/0.668 | -0.046 [-0.138, +0.006] | -0.041 | -0.062 | 16/26 | no |
| **IL+CL local** | 146 | 0.901/0.771 | **+0.086** [+0.044, +0.129] | +0.031 | +0.000 | 110/36 | yes |
| **CD+ID domain** | 73 | 0.584/0.601 | **-0.049** [-0.132, -0.016] | -0.046 | -0.061 | 26/47 | no |

## Primary contrast (local − domain) med ΔAUROC

**+0.136** [+0.077, +0.218]  lock CI>0: **PASS**

## Hypothesis checklist

- ΔAUROC_local > 0: **PASS**
- ΔAUROC_domain ≉ locked >0: **PASS**
- (CL+IL)−(CD+ID) > 0: **PASS**
