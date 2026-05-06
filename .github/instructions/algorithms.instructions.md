---
applyTo: "**/*.py,**/*.ipynb"
---
# Algorithm Implementation Rules

These rules apply whenever implementing logic derived from an academic paper,
arXiv preprint, or novel algorithm not already in a well-known library.

## Citation and Attribution

- Add a citation header at the top of every function or class that implements
  paper-derived logic, in this exact format:
  ```python
  # Based on: Author et al. (YYYY). Title. arXiv:XXXX.XXXXX
  ```
- If the implementation covers only a specific section or equation, note it:
  ```python
  # Implements Section 3.2 / Eq. (4) of the above.
  ```
- For algorithms adapted from published (non-arXiv) sources, cite in the same
  docstring format with journal/conference name in place of the arXiv ID.

## Fidelity to the Paper

- Keep variable names close to the paper's notation on first implementation;
  add a comment mapping them to plain-English names if they are cryptic.
- Do **not** over-generalise on first pass — implement the paper's described
  scope and note assumptions explicitly as `# Assumption: ...` comments.
- When the paper specifies a hyperparameter default, preserve it as a keyword
  argument default and note the paper's value in the docstring.
- If you deviate from the paper (e.g., to handle edge cases not covered),
  mark the deviation with `# NOTE: deviation from paper — <reason>`.

## Complexity and Correctness

- For any algorithm with non-obvious time/space complexity, add a one-line
  comment above the implementation: `# O(n log n) — see Section 4.1`
- Prefer numerically stable formulations; note if you chose a stable variant
  over the paper's exact formula.
- Include at least one sanity-check assertion or `# Invariant:` comment at
  key loop boundaries for algorithms with subtle correctness conditions.

## Prototype vs. Production Distinction

Mark the maturity of an implementation at the module/cell level:
```python
# STATUS: prototype — matches paper but not optimised for production scale
# STATUS: production — validated against paper benchmark, AML-ready
```
Do not silently promote prototype code to production training jobs.
