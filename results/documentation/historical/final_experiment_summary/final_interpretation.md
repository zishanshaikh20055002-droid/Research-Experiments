# Final Interpretation

These summaries use the fixed selected validation fold and three requested seeds only. No statistical significance test was run.

M4 does not add clear RUL value beyond M2 in this pass: M4 mean MAE is 0.304551 versus M2 0.297749, and M4 mean RMSE is 0.385299 versus M2 0.381213. M4 has a higher mean Macro-F1 than M2, but this should be treated cautiously because the selected fold has no Cage-class training bearing.

M5 improves RUL metrics relative to M4 in this pass: M5 mean MAE is 0.289684 versus M4 0.304551, while M5 mean RMSE is 0.360542 versus M4 0.385299. This does not make M5 the preferred overall model because M5 has lower mean Macro-F1 than M4 and worse strict monotonicity and adjacent-change diagnostics.

M3 remains the strongest RUL model by mean MAE (0.251374), mean RMSE (0.312603), and mean R2 (-0.228620) among these five models.

Strict monotonicity metrics do not improve with the monotonicity-regularized models in these saved test predictions. The lowest mean violation rate is M4, so the monotonicity term should not be described as reducing test-time violations here.

Spearman trend correlation is a trend-following diagnostic, not proof of physics consistency. It can give a different view than strict adjacent violation rate because it measures the overall ranking of predictions over time rather than every local increase.

The lowest mean adjacent prediction change is M1. Use this as a smoothness diagnostic only; smoother predictions are not automatically more accurate.

Fault Macro-F1 remains unreliable as a primary claim because the selected training fold contains no Cage-class bearing. Per-class fault conclusions should therefore be conservative.

Cross-condition performance should be discussed carefully, especially for the 40Hz10kN condition where earlier diagnostics showed the hardest generalization behavior.

Seed variability is material, particularly for M3 Macro-F1 and R2. The paper should emphasize mean +/- sample standard deviation rather than a single best seed.
