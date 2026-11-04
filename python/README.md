# tailor-aoi

Python implementation of **TAILOR**, the age-optimal joint sampling and preemption policy from
*Taming the Heavy Tail: Age-Optimal Preemption* (A. Li, Y. İnce, E. Uysal, IEEE ISIT 2026,
[DOI 10.1109/ISIT62367.2026.11653616](https://doi.org/10.1109/ISIT62367.2026.11653616)).

```bash
pip install "tailor-aoi[plot] @ git+https://github.com/AiminLi-Hi/aoi-preemption-heavy-tail.git#subdirectory=python"
```

```python
from tailor_aoi import ServiceDist, solve, compare
dist = ServiceDist.pareto2(mean=2.0, variance=40.0)
policy = solve(dist, kappa_s=1.0, kappa_p=1.0)
print(compare(dist, 1.0, 1.0, policy=policy))
```

See the [repository README](https://github.com/AiminLi-Hi/aoi-preemption-heavy-tail) for details,
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a tour, and run `pytest tests` to verify.
Please cite the paper if you use this package (`python -m tailor_aoi --cite`).
