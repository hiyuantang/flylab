# Dopamine teaching

FlyLab's `mb-dopamine-v1` is an experimental, minimal mushroom-body learning model. It uses MaleCNS identities and measured connections, with physiology and plasticity assumptions added explicitly. It is not a complete model of dopamine, a reproduction of the Shiu paper, or validated behavioral learning.

## Use

In **Teach the fly**, set strength and choose **Reward** (positive score) or **Punish** (negative score). Pair feedback with the cue or action currently being simulated. A pulse lasts 200 milliseconds of **simulation time**; a paused simulation must Run or Step to deliver it. An active pulse cannot be stacked. Scores range from −1 to +1; zero does nothing.

The first pulse enables the external teaching model. Only the selected pathway's increase above its pre-feedback firing baseline contributes during a pulse. The resulting local dopamine proxy decays afterward, so learning can briefly continue after the pulse ends. Endogenous dopamine modulation outside this teaching protocol is not implemented. **Stop teaching** cancels the pulse and freezes learned weights; neural activity continues. **Save state / Resume saved** preserves weights, traces, pending stimulation, and the random generator. **Reset**, scene changes, and other experiment resets erase learning.

The existing Odor A/B controls share a sensory encoding. They cannot yet support a discriminative A-versus-B conditioning experiment. Local synaptic change does not establish learned walking, successful task behavior, or subjective reward.

## Anatomical scope

| Score | Dopamine population | Assumed compartment | Plastic output |
|---|---|---|---|
| Positive | PAM01, predicted dopamine | γ5 | KCγ → MBON01 |
| Negative | PPL101, predicted dopamine | γ1pedc | KCγ → MBON11 |

Membership requires the annotated type and transmitter. KC membership requires `class=Kenyon_Cell`, a `KCg` type, and acetylcholine. Each plastic edge must already exist, and at least one selected DAN must have measured edges to **both** that KC and the target MBON. No same-side shortcut, invented connection, or universal dopamine target is used. Other dopamine neurons and every original graph edge remain present.

For the installed MaleCNS graph (`2be8452f9f4c02d26b71a821277b1e6513cdefbecf32b735c2dba3af62f6dbab`), population selection gives 44 PAM01 and two PPL101 neurons. Exact eligible-edge counts are calculated from the graph and shown in the UI.

Cell-type/compartment interpretations draw on published mushroom-body anatomy, including other specimens. Body-level connectivity does not locate receptors or prove that a DAN contact and a KC output synapse occupy the same subcellular compartment. The selected PAM01 population includes subtypes; this first model does not distinguish their context-dependent functions. MBON01 also innervates β′2a; restricting plastic presynaptic cells to KCγ narrows, but does not anatomically reconstruct, its γ5 component.

## Dynamics and learning assumptions

1. A score stimulates one DAN population with independent Poisson voltage events at `120 × abs(score)` Hz for 200 ms. It uses the existing paper engine's `250 × physiology.gain` mV event jump, leaving refractory periods intact. This is an artificial stimulation protocol, not a measured conversion from reward units to dopamine concentration. Small scores have stochastic outcomes. A partial final command interval averages the remaining exposure to preserve expected event count.
2. At each muscle-command boundary, the existing 50 ms filtered firing rates supply KC and DAN activity. KC rates divided by 100 Hz are clipped to [0, 1]. During the feedback pulse, selected DAN rates minus their pre-feedback rates are divided by 100 Hz and clipped to [0, 1]; DAN output gains also apply. Other DANs contribute zero to this external teaching channel. This readout saturation does not clip neural firing itself.
3. For each eligible KC→MBON edge, a normalized weighted average of its supporting DAN rates uses measured DAN→KC synapse counts. A 200 ms low-pass filter defines a dimensionless local dopamine proxy. After a pulse its input is zero and the proxy decays. This isolates artificial reinforcement from the current brain model's high tonic firing; baseline subtraction and the selected-channel gate are assumptions, not a calibrated reward-prediction-error model. Feedback at startup has a zero baseline and can include startup transients; pairing after a settling period is preferable. Counts are an assumed routing weight, not measured release or concentration.
4. KC activity has a 500 ms eligibility trace, observed even while learning is off. Local dopamine and recent KC activity jointly cause depression:

   `factor *= exp(-1/s × dt × KC_trace × local_dopamine)`

   Factors are bounded to [0.2, 1]. Both valences depress connections onto **different output neurons**. Negative feedback does not create negative dopamine. There is no potentiation, extinction rule, receptor-specific response, reward-prediction-error estimator, or dopamine diffusion in this version.

The 200/500 ms time constants, rate scale, learning rate, bounds, pulse duration, and strength are versioned engineering choices. Updates use simulated seconds, with effective weights applied to the next neural command interval. They do not depend on wall-clock speed or skipped steps. This boundary approximation is not sub-millisecond plasticity integration.

## Execution and persistence

The canonical CSR weights, integer synapse counts, neuron IDs and topology are immutable. Learned factors separately modify only eligible effective CPU outgoing weights and GPU incoming weights. MPS retains its explicitly selected neural precision. Small learning traces and factors use CPU float64, and only selected weights are uploaded; existing neural-rate readback is reused. No dense whole-brain matrix is created and no neurons or integration steps are removed.

`flylab-live-v3` additionally stores the model profile, anatomical mapping hash, factors, traces, pre-feedback firing baseline, pulse state, and last feedback event. Older checkpoints load with learning off. Device migration reapplies learned factors; invalid learning state cannot overwrite a checkpoint. Feedback commands include the UI episode number so a stale tab cannot reward an experiment that has been reset.

## Sources

- [MaleCNS downloads](https://male-cns.janelia.org/download/): measured connections, IDs and predicted transmitter annotations.
- [Li et al., 2020](https://elifesciences.org/articles/62576): mushroom-body compartments, DAN/KC/MBON organization, PAM01 and MBON identities, and subtype complexity.
- [Aso and Rubin, 2016](https://elifesciences.org/articles/16135): cell-type-specific dopamine reinforcement and local plasticity; supports a restricted learning mechanism, not a universal signed dopamine signal.
- [Aso et al., 2014](https://elifesciences.org/articles/04580): mushroom-body output pathways and behavioral valence.

Validation should distinguish correct software routing, induced synaptic plasticity, successful conditioning, and biological fidelity. Only the first two are established by the implementation tests.
