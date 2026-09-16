# Wing aerodynamics and flight experiments

FlyLab's **Flight lab** provides an isolated, force-driven wing environment with a calibrated wingbeat and an explicitly labeled **engineering hover assist**. The current release passes one-second free-flight checks under small initial disturbances. It does not train a flight policy, demonstrate ground takeoff, or establish biological fidelity.

## Use the environment

1. Open **Flight lab**. Run the default **40 ms** tethered experiment to inspect individual wingbeats.
2. Select **Free release**. The default becomes **Engineering hover assist** and a **1 second** flight check.
3. Run the test. The body first spends 80 ms on an explicit tether while its wings spin up, then the tether disengages. Playback shows the subsequent free motion.
4. Compare **Fixed wingbeat · no balance feedback**, **Vacuum**, or **Wings stationary**. Hover assist requires its calibrated wingbeat; fixed-wingbeat mode allows frequency and amplitude changes.

The replay contains actual recorded body and wing poses. Longer tests show the body trajectory at sampled instants; use the 40 ms test to resolve individual wingbeats. A one-second result does not establish indefinite hovering. The pass label applies only to the selected duration and requires, throughout that interval:

- No contact with the floor or other environment geometry.
- Body tilt no greater than 15°.
- Root height error no greater than 1 mm from the 10 mm release height.
- Horizontal root displacement no greater than 5 mm.

These are engineering acceptance thresholds, not biological reference ranges. The test begins in the air and uses a known initial location and orientation. The controller observes simulator state without sensor noise or latency. Ground launch, larger disturbances, wind robustness and longer flight horizons require separate validation.

## Physics and actuation

The fluid approximation is MuJoCo's native stateless ellipsoid model, developed for the published [FlyBody whole-body locomotion simulator](https://www.nature.com/articles/s41586-025-09029-4). It provides an efficient approximation of drag, viscous resistance, added-inertia effects, Kutta lift and Magnus lift without a spatial fluid grid. See [MuJoCo's fluid equations](https://mujoco.readthedocs.io/en/stable/computation/fluid.html).

Each wing has one massless, non-colliding aerodynamic ellipsoid. Other body segments use the native inertia-based fluid model. Existing wing collision patches remain active; they do not each receive an additional fluid model. All 104 generalized velocity coordinates, 85 animal bodies and 196 original muscle channels remain in the mechanical model. Six bounded position servos drive the wing hinges. No neural controller executes in this environment.

| Parameter | Current configuration |
| --- | --- |
| Rig / controller versions | `wing-fluid-rig-v2` / `wing-hover-assist-v1` |
| Units | mm, g, s; forces µN; torques µN·mm; mechanical power µW |
| Animal mass / weight | 1.05831 mg / 10.3820211 µN; excludes the fixed tether fixture |
| Gravity | 9,810 mm/s² downward |
| Air density | 1.225 kg/m³ = 1.225×10⁻⁶ g/mm³ |
| Dynamic viscosity | 1.81×10⁻⁵ Pa·s = 1.81×10⁻⁵ g/(mm·s) |
| Wing ellipsoid semiaxes | 0.54, 1.2, 0.01 mm |
| Fluid coefficients | 1.0, 0.5, 1.5, 1.7, 1.0; retained starting prior from FlyBody |
| Integration | Native MuJoCo `implicitfast`, float64, 50 µs per physical step |
| Wingbeat frequency | 275 Hz |
| Stroke amplitude / mean angle | 1.2033435 / −0.25 rad |
| Mean elevation / stroke-plane pitch | 0.2858375 / 0.0791347 rad |
| Feathering | 0.8 × tanh(3 cos(phase)) rad |
| Wing actuators | Stiffness 100 µN·mm/rad; damping 0.01 µN·mm·s/rad; torque bounded to ±30 µN·mm |

The flight-only hinges use stroke as the outer rotation, followed by elevation and feathering. This avoids placing the large stroke angle in the middle of the three-axis rotation, where the original order approached an Euler singularity. The source meshes, masses, original leg mechanics and checkpoint compatibility inputs remain unchanged. Wing orientation, rigid geometry, hinge location and servo parameters are engineering approximations; the servos do not reconstruct asynchronous flight muscles.

No running step assigns a body pose, body velocity or desired flight trajectory. After the declared tether release, all movement comes from wing actuator torques, native aerodynamic forces, gravity and contact mechanics. Optional robustness-test initial conditions are applied once at the release boundary. The body-force override arrays remain zero during the tests.

### Internal calibration

The trim fit varies only stroke-plane pitch, mean elevation and stroke amplitude. It minimizes mean horizontal force, vertical force minus animal weight, and pitch moment about the whole-animal center of mass. Air properties, fluid coefficients, body mass and geometry are held fixed. Each objective evaluates 80 ms of tethered motion and averages the final 11 complete wingbeats.

The production preset gives about **10.3809 µN** mean upward force, **−0.00065 µN** horizontal force and **0.0033 µN·mm** pitch moment about the center of mass. This is internal balance in the simulator, not calibration against experimentally measured fly forces. The repeatable fit and its parameter evaluations are recorded in [wing-trim-calibration.json](results/wing-trim-calibration.json).

### Balance feedback

The hover assist combines the fixed wingbeat generator with feedback from root position, orientation, linear velocity and angular velocity. A moving average spanning approximately one wingbeat suppresses body vibration at the wingbeat frequency. Bounded proportional, derivative and integral terms adjust common elevation and amplitude, while left/right differences adjust roll and yaw. Commands are updated at every physical step and are subject to the same actuator limits as the fixed pattern.

This controller is a transparent engineering baseline. It is neither a learned policy nor a model of the fly's nervous system. The separation between a fast wingbeat generator and slower steering is also used in FlyBody's learned flight-control hierarchy, but this controller's gains and behavior are specific to this rig.

## Measurements and validation

[The full validation record](results/wing-fluid-validation.json) contains configuration, source hashes, initial conditions, timestep, force averages, pose errors, contacts and runtime for every case. All eight one-second assisted cases passed: nominal release, a halved timestep, ±5° initial pitch, ±5° initial roll and ±25 mm/s initial forward velocity.

| Result across the eight assisted cases | Measured limit |
| --- | --- |
| Largest height error | Less than 0.23 mm |
| Largest body tilt | Less than 8° |
| Largest horizontal displacement | Less than 3.6 mm |
| Environment contacts | Zero |
| Applied body/generalized force overrides | Zero |

The fixed-wingbeat comparison fails the free-flight check: over 200 ms it reaches about 77° tilt and 1.8 mm height error. Hover assist in vacuum also fails; aerodynamic force is exactly zero. Stationary wings do not generate sustained hovering lift. Halving and quartering the timestep change the tethered mean upward force by approximately 0.7% and 1.0%. The halved-step free-flight case also passes, with different detailed trajectories; this is a numerical robustness check, not evidence of exact convergence.

**Measurement conventions matter.** Force and signed actuator power use MuJoCo's pre-integration fields, with a separate force timestamp. Pitch torque is transferred from the root reference point to the whole-animal center of mass at that same force state. Displayed pose and velocity are the subsequent integrated state. Averages cover complete wingbeats, with fractional edge-step weights. Signed mechanical servo power is not metabolic power. Reported pre-integration force averages are not reconstructed discrete-step impulses; implicit integration and timestep affect their agreement with measured body momentum.

Wing/body contacts still occur in the assisted trajectories and are reported explicitly, including their frequency in the raw results. They are retained in the physics rather than disabled to obtain flight. Their anatomical realism and the wing geometry require further validation before using this environment to make biological claims. Wake capture, wing-wing aerodynamic interaction, flexible wings, ground effect and detailed thoracic mechanics are outside the approximation.

## Runtime and reproducibility

A one-second assisted release plus 80 ms of spin-up takes roughly 20–25 seconds on the development machine. It retains all physical substeps. Native timer profiling attributes most of the cost to the implicit integration stage, with collision detection a smaller component. Python-only optimization would provide limited benefit. This is an efficient fluid approximation relative to a resolved flow simulation, but the full articulated rig is not real-time. See [the runtime profile](results/wing-runtime-profile.json); its timers overlap and must not be added together.

Rigid-body and aerodynamic simulation run on CPU in float64. Existing transformer training remains on Apple MPS GPU in float32. Flight experiments do not replace the loaded leg policy, create a checkpoint, or change the existing 84-command leg action schema.

```sh
PYTHONPATH=backend uv run python scripts/calibrate_flight.py
PYTHONPATH=backend uv run python scripts/validate_flight.py --full --workers 4
PYTHONPATH=backend uv run python scripts/profile_flight.py
PYTHONPATH=backend uv run pytest tests/test_flight.py tests/test_policy_bc.py tests/test_policy_checkpoint.py tests/test_policy_model.py tests/test_api.py -q
(cd frontend && npm run build)
```

The API bounds requests to 40 ms, 200 ms or one second and serializes experiments. If a client disconnects, its computation retains the lock until completion. Solver warnings, nonfinite states and unexpected simulation-clock resets abort the experiment. Tests cover trim balance, whole-cycle averaging, center-of-mass torque, solver-failure detection, free-flight support, vacuum failure, model isolation, input validation and cancellation handling. The combined regression run passed **54 tests with three environment-dependent skips**. The frontend production build passed with its existing bundle-size warning. Isolated Chrome verified real assisted-flight and vacuum requests, pause/scrub playback, and 1440 px / 800 px layouts without browser errors or horizontal overflow. See the [UI validation record](results/wing-ui-validation.json), [desktop view](screenshots/flight-hover.png) and [compact view](screenshots/flight-hover-compact.png).

## Next learning stage

The next task is gesture-controlled flapping using a new, versioned wing-command schema and held-out gesture tests. The current trained network emits leg commands and cannot operate these wing servos. The first learned behavior can enable or modulate the declared wingbeat generator; the engineering assist remains separately identified. Learned flight control, takeoff and more general trajectories need their own training objectives and physical rollout evaluations.
