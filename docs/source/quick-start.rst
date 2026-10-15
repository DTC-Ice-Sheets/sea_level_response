Quick Start
===========

The package provides two CLI entry points, one to calculate global sea-level response given an input mass balance dataset
and another to extract a sea-level time series at a specific location.

Run either with ``--help`` to see all options.

.. code-block:: bash

   python -m sea_level_response.selrem_entrypoint --help
   python -m sea_level_response.slr_ts_entrypoint --help

Global SLR computation
----------------------

:mod:`sea_level_response.selrem_entrypoint` computes global-mean or gridded annual sea-level response
from a vertical mass balance (VMB) GeoZarr dataset:

.. code-block:: bash

   python -m sea_level_response.selrem_entrypoint \
       --mass-balance-geozarr /path/to/mass_balance.zarr \
       --start-time 2000-01-01 \
       --end-time 2020-01-01 \
       --analysis-mode annual \
       --output-path /path/to/output.zarr

Set ``--analysis-mode global`` for a single global-mean value instead of a gridded annual series.
Use ``--coarsen-factor`` to spatially coarsen gridded inputs before computation.

Location time-series extraction
-------------------------------

:mod:`sea_level_response.slr_ts_entrypoint` extracts a sea-level time series at a single location
from the annual, gridded output produced above:

.. code-block:: bash

   python -m sea_level_response.slr_ts_entrypoint \
       --annual-slr-gz-path /path/to/output.zarr \
       --latitude 69.2 \
       --longitude -49.9 \
       --output-path /path/to/timeseries.json

The result is written as JSON containing the extracted time series for the given coordinates.
