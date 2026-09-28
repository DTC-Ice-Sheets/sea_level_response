.. DTC Python Template documentation master file, created by
   sphinx-quickstart on Tue Jan 21 10:28:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DTC Ice Sheets: Sea Level Response
===================================

TODO - write some Documentation here... 

Quick Start
===========

The package provides two CLI entry points, one to calculate global sea-level rise given an input mass balance dataset 
and another to extract a sea-level time series at a specific location.

Run either with ``--help`` to see all options.

Global SLR computation
----------------------

:mod:`sea_level_response.selrem_entrypoint` computes global-mean or gridded annual sea-level rise
from a vertical mass balance (VMB) GeoZarr dataset:

.. code-block:: bash

   python -m sea_level_response.selrem_entrypoint \
       --mass-balance-geozarr /path/to/mass_balance.zarr \
       --start-time 2000-01-01 \
       --end-time 2020-01-01 \
       --analysis-mode annual \
       --output-path /path/to/output.zarr

Set ``--analysis-mode global`` for a single global-mean value instead of a gridded annual series.
Use ``--coarsen-factor`` to spatially coarsen gridded inputs, and ``--scaling-factor`` to scale the
input mass balance values before computation.

Location time-series extraction
--------------------------------

:mod:`sea_level_response.slr_ts_entrypoint` extracts a sea-level time series at a single location
from the annual, gridded output produced above:

.. code-block:: bash

   python -m sea_level_response.slr_ts_entrypoint \
       --annual-slr-gz-path /path/to/output.zarr \
       --latitude 69.2 \
       --longitude -49.9 \
       --output-path /path/to/timeseries.json

The result is written as JSON containing the extracted time series for the given coordinates.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   API <api.rst>
