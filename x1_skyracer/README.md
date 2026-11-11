# X1 SkyRacer (x1_skyracer)

The X1 SkyRacer is Mac McAlpine's open flying car race design for X1 Racing: two seats, a series-hybrid propane
powertrain, eight lift fans in four coaxial ducts, two ducted pushers and no control surfaces, raced unmanned at
3,500 lb with 500 lb of ballast in the seats. No SkyRacer has been built yet, so this is a concept-level JSBSim model,
not test data. It is published here with the designer's permission (`NOTICE`).

## The design, from its source

- [The X1 SkyRacer page](https://buildflyingcars.com/x1-skyracer/): the design, its numbers, updates and downloads
- [3D viewer](https://buildflyingcars.com/x1-skyracer/viewer/)
- [Design repository](https://github.com/projectmaximumboost/x1skyracer): the script that builds the design, its
  studies and the power plant models
- [Flight model kit](https://buildflyingcars.com/x1-skyracer/X1_SkyRacer_PteroSim.zip): the designer's JSBSim, PX4 and
  ArduPilot kit
- [X1 Racing Rule Book](https://x1racing.com/rule-book)
- [X1 Racing](https://x1racing.com) and [Design your own](https://buildflyingcars.com/x1-racer)

The meshes and the rotor stations follow design revision 35396c71 (10 Nov 2026, simulator cockpit on the fan-tilt
airframe): the fuselage is open under the canopy from the dash to the firewall, the canopy edge is one shared
60-station loop, and the cabin has the tub, decks, dash, centre screen, console and stick. Rear ducts cant 10°
(thrust forward), the pushers sit at y ±0.45 m and 0.80 m with the exhaust 10° down, and the wings in the mesh are
at +5°. Mass, power and the fan tables are the October flight model; the aero tables are still that AVL solution
about its original reference point and do not add the 5° incidence. The ArduPilot hover split is still the October
trim. Go to the links above for the current design.

```
x1_skyracer.xml  main file
Metrics.xml      reference areas; structural frame = on the ground under the craft centre, x aft, y right, z up
Mass.xml         dry 2,800 lb from the designer's mass list; 500 lb of ballast 0.28 m above the new seat pans
Gear.xml         four wheels under the sponsons; hard points on the nose, tail, ducts and canopy
Propulsion.xml   eight lift fans, two pushers, three propane tanks (200 lb)
Aero.xml         AVL derivatives about the October reference point, then a stalled flat plate
Engines/         fan tables (with the lift fans' in-plane force) and the motors
Systems/         power.xml: generator, accumulator, propane burn, power priority and motor drives
Controls.xml     the ten autopilot outputs to the ten motors (sidecar)
Sensors.xml      imu, gps, barometer, temperature, airspeed, driver's camera (sidecar)
Visual.xml       meshes and the fans' spin (sidecar)
firmwares/       ArduPlane parameters and motor script, PX4 airframe 22008
LICENSE, NOTICE  GPL-3.0 for the flight model, the meshes' terms, sources
```

## Run it

Under ArduPlane SITL, with PteroSim's physics at 1 kHz. The motor script has to be in `scripts/` under SITL's working
directory:

```
mkdir -p scripts && cp firmwares/ardupilot_x1_skyracer_motors.lua scripts/
sim_vehicle.py -v ArduPlane --model JSON:<host> -w --add-param-file=firmwares/ardupilot_x1_skyracer.param
```

Fly AUTO missions or the Q modes. With `Q_ENABLE 2` every AUTO leg is hovered and the pushers give the forward
acceleration. Do not use GUIDED: it flies the fixed-wing law, and this craft has no surfaces for that.

PX4: `firmwares/px4_x1_skyracer` is airframe 22008; install it as a SITL airframe and start with
`PX4_SYS_AUTOSTART=22008`. It is a Standard VTOL that stays in multicopter mode.

The model needs PteroLabs' JSBSim fork (PteroLabsAI/jsbsim) for the lift fans' in-plane force table. Stock JSBSim logs
"Unknown table type" for that table and applies no in-plane force.

Check the meshes with `python3 x1_skyracer/tests/test_visual_assets.py` from the repository root.

## Known limits

- The fan blades, the motors and the power electronics are sized for this model; they are not X1's. The blades are
  twisted NACA 4409 sections, 13 and 17 per lift pair and 11 per pusher (`Engines/`).
- The aerodynamics come from a vortex-lattice model and handbook drag, and the airframe is unstable in pitch: it flies
  on the autopilot, with no Manual pilot in `Controls.xml`.
- The lift fans' ram drag grows with speed, so level speed stays well below the design's published estimates, which
  come from a different power model.
- The attitude gains in `firmwares/` are worked out from the design's inertia and fan lag, not tuned on a craft.
- Thruster axes follow revision 35396c71, but the AVL tables, the inertia and the ArduPilot hover split were not recomputed for the cant, the pusher tilt or the moved seats. The meshes were checked with `tests/test_visual_assets.py` and a software rasterizer, not inside PteroSim.
