# Monotonicity Comparison

Offline analysis using saved test prediction CSVs only. Predictions are grouped strictly by bearing and sorted chronologically by sequence end position. True RUL is not used for violation calculation.

## Overall

| model | violation_rate | mean_violation_magnitude | violating_transitions | valid_adjacent_transitions |
| --- | --- | --- | --- | --- |
| M1 | 0.463074 | 0.00657573 | 1461 | 3155 |
| M2 | 0.478605 | 0.0121471 | 1510 | 3155 |
| M3 | 0.484628 | 0.0156894 | 1529 | 3155 |
| M5 | 0.494136 | 0.0157764 | 1559 | 3155 |

## By Condition

| model | condition | violation_rate | mean_violation_magnitude | violating_transitions | valid_adjacent_transitions |
| --- | --- | --- | --- | --- | --- |
| M1 | 35Hz12kN | 0.354305 | 0.0186767 | 107 | 302 |
| M1 | 37.5Hz11kN | 0.449095 | 0.0104625 | 397 | 884 |
| M1 | 40Hz10kN | 0.486034 | 0.00361036 | 957 | 1969 |
| M2 | 35Hz12kN | 0.440397 | 0.0323081 | 133 | 302 |
| M2 | 37.5Hz11kN | 0.467195 | 0.0147611 | 413 | 884 |
| M2 | 40Hz10kN | 0.489589 | 0.00824569 | 964 | 1969 |
| M3 | 35Hz12kN | 0.486755 | 0.0353008 | 147 | 302 |
| M3 | 37.5Hz11kN | 0.477376 | 0.0241717 | 422 | 884 |
| M3 | 40Hz10kN | 0.487557 | 0.00895768 | 960 | 1969 |
| M5 | 35Hz12kN | 0.476821 | 0.040493 | 144 | 302 |
| M5 | 37.5Hz11kN | 0.489819 | 0.0259586 | 433 | 884 |
| M5 | 40Hz10kN | 0.49873 | 0.00766227 | 982 | 1969 |

## By Bearing

| model | bearing_id | condition | violation_rate | mean_violation_magnitude | violating_transitions | valid_adjacent_transitions |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Bearing1_3 | 35Hz12kN | 0.385135 | 0.0166716 | 57 | 148 |
| M1 | Bearing1_4 | 35Hz12kN | 0.375 | 0.0190845 | 42 | 112 |
| M1 | Bearing1_5 | 35Hz12kN | 0.190476 | 0.0308216 | 8 | 42 |
| M1 | Bearing2_3 | 37.5Hz11kN | 0.458891 | 0.0138603 | 240 | 523 |
| M1 | Bearing2_4 | 37.5Hz11kN | 0.4375 | 0.0071308 | 14 | 32 |
| M1 | Bearing2_5 | 37.5Hz11kN | 0.43465 | 0.00508621 | 143 | 329 |
| M1 | Bearing3_3 | 40Hz10kN | 0.534626 | 0.0066459 | 193 | 361 |
| M1 | Bearing3_4 | 40Hz10kN | 0.486037 | 0.00287188 | 731 | 1504 |
| M1 | Bearing3_5 | 40Hz10kN | 0.317308 | 0.00221545 | 33 | 104 |
| M2 | Bearing1_3 | 35Hz12kN | 0.466216 | 0.029307 | 69 | 148 |
| M2 | Bearing1_4 | 35Hz12kN | 0.410714 | 0.0284666 | 46 | 112 |
| M2 | Bearing1_5 | 35Hz12kN | 0.428571 | 0.0536291 | 18 | 42 |
| M2 | Bearing2_3 | 37.5Hz11kN | 0.479924 | 0.0184609 | 251 | 523 |
| M2 | Bearing2_4 | 37.5Hz11kN | 0.40625 | 0.0336352 | 13 | 32 |
| M2 | Bearing2_5 | 37.5Hz11kN | 0.452888 | 0.00688184 | 149 | 329 |
| M2 | Bearing3_3 | 40Hz10kN | 0.440443 | 0.0106326 | 159 | 361 |
| M2 | Bearing3_4 | 40Hz10kN | 0.507314 | 0.00752977 | 763 | 1504 |
| M2 | Bearing3_5 | 40Hz10kN | 0.403846 | 0.0122155 | 42 | 104 |
| M3 | Bearing1_3 | 35Hz12kN | 0.52027 | 0.0454204 | 77 | 148 |
| M3 | Bearing1_4 | 35Hz12kN | 0.482143 | 0.0284904 | 54 | 112 |
| M3 | Bearing1_5 | 35Hz12kN | 0.380952 | 0.00958524 | 16 | 42 |
| M3 | Bearing2_3 | 37.5Hz11kN | 0.497132 | 0.0335485 | 260 | 523 |
| M3 | Bearing2_4 | 37.5Hz11kN | 0.375 | 0.0418605 | 12 | 32 |
| M3 | Bearing2_5 | 37.5Hz11kN | 0.455927 | 0.00650338 | 150 | 329 |
| M3 | Bearing3_3 | 40Hz10kN | 0.473684 | 0.0120977 | 171 | 361 |
| M3 | Bearing3_4 | 40Hz10kN | 0.495346 | 0.00821156 | 745 | 1504 |
| M3 | Bearing3_5 | 40Hz10kN | 0.423077 | 0.00938754 | 44 | 104 |
| M5 | Bearing1_3 | 35Hz12kN | 0.493243 | 0.0517623 | 73 | 148 |
| M5 | Bearing1_4 | 35Hz12kN | 0.491071 | 0.0328511 | 55 | 112 |
| M5 | Bearing1_5 | 35Hz12kN | 0.380952 | 0.0153463 | 16 | 42 |
| M5 | Bearing2_3 | 37.5Hz11kN | 0.516252 | 0.0362052 | 270 | 523 |
| M5 | Bearing2_4 | 37.5Hz11kN | 0.375 | 0.0291297 | 12 | 32 |
| M5 | Bearing2_5 | 37.5Hz11kN | 0.458967 | 0.00738481 | 151 | 329 |
| M5 | Bearing3_3 | 40Hz10kN | 0.481994 | 0.0116152 | 174 | 361 |
| M5 | Bearing3_4 | 40Hz10kN | 0.509973 | 0.00680573 | 767 | 1504 |
| M5 | Bearing3_5 | 40Hz10kN | 0.394231 | 0.00690996 | 41 | 104 |
