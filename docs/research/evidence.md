# Research and evidence

This file records methods relevant to the camera-first prototype. **No source below validates a universal camera threshold for entry safety or imminent collapse.** Thresholds in structural-health research depend on the measured quantity, building type, geometry, design level, damage state, and calibration.

## Sources

1. [Hsu and Kuo, *A Stand-Alone Smart Camera System for Online Post-Earthquake Building Safety Assessment* (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7349837/). A single-board computer, camera, two accelerometers, reference targets, and data fusion measured interstory drift in shaking-table tests of a scaled six-story steel building. Damage levels were compared with building-specific HAZUS-MH drift thresholds and laboratory displacement sensors. Camera-only estimates sometimes understated damage; fused estimates matched the reference damage levels in that experiment. **Use:** a concrete method and validation model. **Limit:** its thresholds and results cannot be transferred directly to an external Pi tripod, ultrasonic sensor, cardboard model, or an arbitrary building.
2. [Cheng, Tsai, and Huang, *Vision-based algorithm for structural response measurement using movable camera and damage localization* (2024)](https://scholars.lib.ntu.edu.tw/entities/publication/c4386efe-0e7c-4dc4-827c-c016cfcf8fb9). The authors combine optical flow, camera pose estimation, and motion compensation, with experimental validation on a frame structure. **Use:** camera movement needs explicit treatment in the coupled and noisy demo cases. **Limit:** a published compensation method is not proof that this team's implementation works.
3. [Komati and Nagmode, *Amplifying Imperceptible Variations in Video Sequences for Time-Varying Process Analysis* (2023)](https://iieta.org/journals/ts/paper/10.18280/ts.400607). Region-of-interest video magnification reveals subtle movement and estimates vibration frequency in a car-engine experiment. **Use:** a candidate way to reveal or analyze small motion. **Limit:** it does not validate building damage, safe entry, or collapse prediction.
4. [USGS Structural Health Monitoring FAQ](https://earthquake.usgs.gov/monitoring/nsmp/buildings/faq.php). USGS describes assessing potential damage using interstory drift ratios and other structural parameters, supported by dense instrumented measurements and ambient-vibration baselines. **Use:** identify meaningful structural quantities and why baseline and sensors matter. **Limit:** this team's portable camera is not equivalent to an instrumented building.
5. [Applied Technology Council ATC-20](https://j6.atcouncil.org/atc-20) and [FEMA P-2055-1](https://atcouncil.org/docman/atc-publications/328-femap-2055-1-guidance-or-accelerated-building-reoccupancy-2023/file). Qualified evaluators use rapid and detailed post-earthquake procedures to decide occupancy or restricted use. **Use:** define the handoff to human assessment. **Limit:** this prototype cannot issue an ATC-20 placard or entry clearance.

## Validation questions for the implementation owner

- What physical motion is measured: pixels, relative displacement, frequency, or a calibrated interstory drift ratio?
- Which features belong to the target, which belong to a fixed reference, and how is camera movement estimated?
- What independent measurement supplies ground truth for the cardboard model?
- How do frame rate, lighting, distance, vibration frequency, blur, occlusion, and camera shake affect errors?
- What does an ultrasonic sensor add in this geometry? Document the exact model and compare its readings against camera and ground truth.
- Which conditions must produce **measurement unreliable** rather than a confident hazard label?
- Are any thresholds experimentally calibrated for this specific model, and are they clearly labeled as **demo thresholds**?

Only after those measurements are validated should the team discuss broader structural inference with an engineer. The 2020 study's numerical drift limits apply to its specified steel-building category and are not defaults for this project.
