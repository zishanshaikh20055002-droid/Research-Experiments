# Fast-Track Comparison

| Model | MAE | RMSE | R2 | Fault Macro-F1 | Mono Violation |
| --- | --- | --- | --- | --- | --- |
| M1 | 0.306988 | 0.401279 | -1.00812 | nan | nan |
| M2 | 0.313711 | 0.393294 | -0.928997 | 0.11245 | nan |
| M3 | 0.271602 | 0.343606 | -0.472377 | 0.0689065 | 0.484628 |
| M5 | 0.288149 | 0.360968 | -0.62493 | 0.0710533 | 0.494136 |

## Per-Condition RUL

| Model | condition | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| M1 | 35Hz12kN | 0.102825 | 0.143537 | 0.71516 |
| M1 | 37.5Hz11kN | 0.156241 | 0.208683 | 0.453448 |
| M1 | 40Hz10kN | 0.406371 | 0.48537 | -1.89144 |
| M2 | 35Hz12kN | 0.126578 | 0.175899 | 0.572237 |
| M2 | 37.5Hz11kN | 0.183644 | 0.237014 | 0.294976 |
| M2 | 40Hz10kN | 0.401157 | 0.467039 | -1.67717 |
| M3 | 35Hz12kN | 0.171592 | 0.212379 | 0.376415 |
| M3 | 37.5Hz11kN | 0.173931 | 0.218446 | 0.401112 |
| M3 | 40Hz10kN | 0.331003 | 0.401237 | -0.97593 |
| M5 | 35Hz12kN | 0.172407 | 0.206584 | 0.409977 |
| M5 | 37.5Hz11kN | 0.171405 | 0.20701 | 0.462176 |
| M5 | 40Hz10kN | 0.358561 | 0.427998 | -1.24829 |

- Best numeric RUL by RMSE: `M3`
- Best monotonicity: `M3`
- Best fault Macro-F1: `M2`

Known limitation: the selected split has no Cage-class training bearing; Cage classification is not learnable supervised under this protocol.
