# Per-muscle activation comparison

Both hands point; final frame at 500 ms. Actual values are from the original-model diagnostic, not the later three-update training candidate. Values are dimensionless effective-muscle activations on a 0–1 scale, not forces or voltages. Targets are simulated engineering references, not biological recordings.

All-zero command baseline is 0 for every muscle. The 84 rows are the supervised leg channels, not all 196 body actuators. CSV retains the numeric precision stored in the diagnostic JSON; this table rounds to six decimals.

Source: [diagnostic JSON](gesture-long-history-diagnosis.json). Full precision: [CSV](muscle-activation-comparison.csv).

Prefixes: lf/rf = left/right front; lm/rm = left/right middle; lh/rh = left/right hind. Positive/negative identify opposing actuator directions, not negative activation values.

| Muscle | Original model | Target | Target − model |
|---|---:|---:|---:|
| lf_coxa_yaw_positive | 0.000010 | 0.286629 | 0.286619 |
| lf_coxa_yaw_negative | 0.000000 | 0.100000 | 0.100000 |
| lf_coxa_pitch_positive | 0.002105 | 0.102347 | 0.100242 |
| lf_coxa_pitch_negative | 0.003436 | 0.104096 | 0.100660 |
| lf_coxa_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| lf_coxa_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| lf_trochanterfemur_pitch_positive | 0.008144 | 0.100000 | 0.091856 |
| lf_trochanterfemur_pitch_negative | 0.000600 | 0.215974 | 0.215374 |
| lf_trochanterfemur_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| lf_trochanterfemur_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| lf_tibia_pitch_positive | 0.000000 | 0.152805 | 0.152805 |
| lf_tibia_pitch_negative | 0.004675 | 0.100000 | 0.095325 |
| lf_tarsus1_pitch_positive | 0.000000 | 0.100000 | 0.100000 |
| lf_tarsus1_pitch_negative | 0.000000 | 0.253956 | 0.253956 |
| lm_coxa_yaw_positive | 0.002154 | 0.100000 | 0.097846 |
| lm_coxa_yaw_negative | 0.000000 | 0.233926 | 0.233926 |
| lm_coxa_pitch_positive | 0.005523 | 0.100000 | 0.094477 |
| lm_coxa_pitch_negative | 0.000003 | 0.140920 | 0.140917 |
| lm_coxa_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| lm_coxa_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| lm_trochanterfemur_pitch_positive | 0.000001 | 0.235124 | 0.235123 |
| lm_trochanterfemur_pitch_negative | 0.004026 | 0.100000 | 0.095974 |
| lm_trochanterfemur_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| lm_trochanterfemur_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| lm_tibia_pitch_positive | 0.000000 | 0.137217 | 0.137217 |
| lm_tibia_pitch_negative | 0.089672 | 0.103256 | 0.013583 |
| lm_tarsus1_pitch_positive | 0.000000 | 0.000000 | 0.000000 |
| lm_tarsus1_pitch_negative | 0.000000 | 0.000000 | 0.000000 |
| lh_coxa_yaw_positive | 0.000401 | 0.429874 | 0.429473 |
| lh_coxa_yaw_negative | 0.000000 | 0.100000 | 0.100000 |
| lh_coxa_pitch_positive | 0.006907 | 0.100000 | 0.093093 |
| lh_coxa_pitch_negative | 0.006730 | 0.300009 | 0.293279 |
| lh_coxa_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| lh_coxa_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| lh_trochanterfemur_pitch_positive | 0.030114 | 0.100000 | 0.069886 |
| lh_trochanterfemur_pitch_negative | 0.000000 | 0.452631 | 0.452631 |
| lh_trochanterfemur_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| lh_trochanterfemur_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| lh_tibia_pitch_positive | 0.000000 | 0.100000 | 0.100000 |
| lh_tibia_pitch_negative | 0.007152 | 0.158665 | 0.151513 |
| lh_tarsus1_pitch_positive | 0.000000 | 0.000000 | 0.000000 |
| lh_tarsus1_pitch_negative | 0.000000 | 0.000000 | 0.000000 |
| rf_coxa_yaw_positive | 0.000000 | 0.286683 | 0.286683 |
| rf_coxa_yaw_negative | 0.000000 | 0.100000 | 0.100000 |
| rf_coxa_pitch_positive | 0.000000 | 0.101825 | 0.101825 |
| rf_coxa_pitch_negative | 0.000911 | 0.103672 | 0.102762 |
| rf_coxa_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| rf_coxa_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| rf_trochanterfemur_pitch_positive | 0.000009 | 0.100000 | 0.099991 |
| rf_trochanterfemur_pitch_negative | 0.000000 | 0.215378 | 0.215378 |
| rf_trochanterfemur_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| rf_trochanterfemur_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| rf_tibia_pitch_positive | 0.000000 | 0.152754 | 0.152754 |
| rf_tibia_pitch_negative | 0.000000 | 0.100000 | 0.100000 |
| rf_tarsus1_pitch_positive | 0.000000 | 0.100000 | 0.100000 |
| rf_tarsus1_pitch_negative | 0.000000 | 0.253965 | 0.253965 |
| rm_coxa_yaw_positive | 0.003926 | 0.100000 | 0.096074 |
| rm_coxa_yaw_negative | 0.000000 | 0.230723 | 0.230723 |
| rm_coxa_pitch_positive | 0.003117 | 0.100000 | 0.096883 |
| rm_coxa_pitch_negative | 0.000000 | 0.147160 | 0.147160 |
| rm_coxa_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| rm_coxa_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| rm_trochanterfemur_pitch_positive | 0.000000 | 0.218180 | 0.218180 |
| rm_trochanterfemur_pitch_negative | 0.025051 | 0.100000 | 0.074949 |
| rm_trochanterfemur_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| rm_trochanterfemur_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| rm_tibia_pitch_positive | 0.000000 | 0.141382 | 0.141382 |
| rm_tibia_pitch_negative | 0.026217 | 0.100036 | 0.073819 |
| rm_tarsus1_pitch_positive | 0.000000 | 0.000000 | 0.000000 |
| rm_tarsus1_pitch_negative | 0.000000 | 0.000000 | 0.000000 |
| rh_coxa_yaw_positive | 0.008394 | 0.109262 | 0.100868 |
| rh_coxa_yaw_negative | 0.000000 | 0.100000 | 0.100000 |
| rh_coxa_pitch_positive | 0.002798 | 0.100000 | 0.097202 |
| rh_coxa_pitch_negative | 0.008809 | 0.140297 | 0.131489 |
| rh_coxa_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| rh_coxa_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| rh_trochanterfemur_pitch_positive | 0.043367 | 0.100000 | 0.056633 |
| rh_trochanterfemur_pitch_negative | 0.000000 | 0.117550 | 0.117550 |
| rh_trochanterfemur_roll_positive | 0.000000 | 0.000000 | 0.000000 |
| rh_trochanterfemur_roll_negative | 0.000000 | 0.000000 | 0.000000 |
| rh_tibia_pitch_positive | 0.000000 | 0.101567 | 0.101567 |
| rh_tibia_pitch_negative | 0.047858 | 0.129527 | 0.081669 |
| rh_tarsus1_pitch_positive | 0.000000 | 0.000000 | 0.000000 |
| rh_tarsus1_pitch_negative | 0.000000 | 0.000000 | 0.000000 |
