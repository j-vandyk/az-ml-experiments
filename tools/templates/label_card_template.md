# Label: `{{OUTCOME}}`

**Family:** {{FAMILY}}
**Source:** {{SOURCE}}
**Owner:** {{OWNER}}
**Card created:** {{TODAY}}

---

## 1. Operational definition

TODO: state the threshold rule in pseudocode. Examples by family:

- **onset** — `label_t = 1 if event_t >= THRESHOLD AND no event in [t-K, t-1]`
- **delta** — `label_t = 1 if indicator_t - indicator_{t-1} < -DELTA_THRESHOLD`
- **percentile** — `label_t = 1 if events_t > q90(events_{country, training_years})`
- **triangulated** — `label_t = 1 if source_A AND source_B`
- **weak-supervision** — list the labeling functions, then `P(label=1 | LFs) > 0.5`

```python
# TODO: paste the actual rule used in the source-pull or feature-engineering notebook
```

## 2. Base rate

| Window | Positives | Total country-years | Base rate |
|---|---|---|---|
| Train (≤2018) | TODO | TODO | TODO |
| Val (2019–2021) | TODO | TODO | TODO |
| Test (2022–2024) | TODO | TODO | TODO |

If base rate < 0.1%, AUPRC is the only meaningful headline metric — see `.claude/metric-interpretation-guide.md` §1.

## 3. Coverage

- **Countries included:** TODO
- **Year range:** TODO–TODO
- **Missing-treatment:** TODO (`0` = assume no event; `NA` = unknown/partial coverage)

## 4. Sensitivity to threshold / peace-spell window

| Variant | Positives | Base rate | AUPRC ceiling (val) |
|---|---|---|---|
| TODO `K=1` | TODO | TODO | TODO |
| TODO `K=2` (chosen) | TODO | TODO | TODO |
| TODO `K=3` | TODO | TODO | TODO |

## 5. Source-agreement rate

(Only relevant for `triangulated` family — delete this section otherwise.)

- **Sources compared:** TODO
- **Agreement rate:** TODO%

## 6. Construct-validity gaps

- TODO

## 7. References

- TODO
