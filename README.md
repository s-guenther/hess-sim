# HESS-EMS-SIM

A simulation framework to test and showcase energy management strategies
(EMS) for hybrid energy storage systems (HESS) defined in
[HESS-EMS](https://github.com/s-guenther/hessems).


## Associated Work

Paper: _forthcoming_ \
EMS Implementations: [HESS-EMS](https://github.com/s-guenther/hessems)

To showcase the EMS, the project makes use of
[ESTSS](https://github.com/s-guenther/estss) and
[hybrid](https://github.com/s-guenther/hybrid).
Note that explicit dependencies on these projects are absent. Utilised 
time series data from `ESTSS` is included in this project, and manually
generated reference data from `hybrid` is also stored here, allowing for result
reproduction without dependencies. The introduced EMS can be operated
independently, and testing and visualizing other time series and EMS without
`ESTSS` and `hybrid` is feasible.


## Overview

The main modules to simulate an EMS are:

- `controllers.py` provides adapters for the EMS defined in
  [HESS-EMS](https://github.com/s-guenther/hessems)
- `storages.py` provides the storage implementation
- `simulate.py` for simulating specific storage settings with specific 
  controllers

To showcase the EMS, the `examples/` folder contains

- `timeseries.py` for data retrieval from
  [ESTSS](https://github.com/s-guenther/estss)
- `reference.py` for data export and import to/from
  [hybrid](https://github.com/s-guenther/hybrid)
- `visualize.py` for standardized simulation results visualization
- `showcase.py` for setting up the concrete EMS simulations and generating 
  the plots

See the source code documentation for further information.

## Requirements

Developed with Python `3.11`. Should work with way older versions since
`3.6` (f-strings) as well.


## Installation

Install manually by cloning the repository, entering it, and build and 
install via the `pyproject.toml`:

```shell
    git clone https://github.com/s-guenther/hessemssim
    cd hessemssim
    python3 -m build
    pip install .
```


## Getting Started

_tdb_


## Contributing

Contributions are welcome! Please feel free to create issues or submit pull
requests for improvement suggestions or code contributions.


## License

Licensed under GPL-3.0-only. This program is free software and can be
redistributed and/or modified under the terms of the GNU General Public License
as published by the Free Software Foundation, version 3. It comes without any
warranty, including the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for details.

Commercial usage under GPLv3 is allowed without royalties or further
implications. However, discussions regarding potential cooperations are
appreciated.


## Author

HESS EMS SIM -
hybrid energy storage system energy management strategies simulation\
Copyright (C) 2024\
Sebastian Günther\
sebastian.guenther@ifes.uni-hannover.de

Leibniz Universität Hannover\
Institut für Elektrische Energiesysteme\
Fachgebiet für Elektrische Energiespeichersysteme

Leibniz University Hannover\
Institute of Electric Power Systems\
Electric Energy Storage Systems Section

https://www.ifes.uni-hannover.de/ees.html


