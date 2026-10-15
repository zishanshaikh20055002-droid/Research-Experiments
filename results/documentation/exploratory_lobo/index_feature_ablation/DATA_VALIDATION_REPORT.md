# Data Validation Report

- POST-HOC AND EXPLORATORY: the fixed models, contrasts, and analyses were run after prior LOBO results had been inspected.
- NPZ X has shape 9,215 x 32; metadata and NPZ row keys agree exactly; metadata target agrees with float32 NPZ RUL to 1e-7.
- All NPZ features and targets are finite; all sequence targets are bounded in [0, 1].
- Built 9,080 sequences independently within bearings with x[t:t+10] -> RUL[t+9].
- The lobo_run_02 fold manifest contains exactly one held-out bearing per fold and is reproduced exactly.
- All engineered features are recording-local; primary ablations use all 32 as prospective inputs.
- Bearing3_4 preserves the source numeric-index gap 1105; it is not synthesized or renumbered.
- No transform uses held-out bearing data: condition encoding is fitted on each fold's training data only; the tree and mean models contain no scaler.
