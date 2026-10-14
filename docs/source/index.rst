.. DTC Python Template documentation master file, created by
   sphinx-quickstart on Tue Jan 21 10:28:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DTC Ice Sheets: Sea-Level Response
==================================


Overview
--------
The **Sea-Level Response** (SLR) module of the DTC Ice Sheets allows users to explore how changes in ice-sheet mass 
affect sea level globally and regionally.

Loss of grounded ice adds water to the ocean, but the resulting sea-level change is not spatially uniform. 
Changes in the gravitational attraction between an ice sheet and the ocean, deformation of the solid Earth (glacial
isostatic adjustment), and redistribution of water due to the Earth's rotation produce characteristic spatial patterns in
sea-level change. Consequently, the sea-level response close to a region of ice loss can differ substantially from 
the global mean response.

This module supports two primary workflows:

* Calculation of gridded sea-level response fields from observed or simulated ice-sheet mass-balance datasets.
* Extraction of sea-level response time series at user-specified locations from precomputed response fields.


Scientific background on SLR calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DTC Ice Sheets uses **SELREM (SEa-Level Response EMulator)** to translate spatially distributed changes in land-ice mass into spatial patterns of sea-level change. It also computes the associated uncertainties for each of these fields.

* The **sea-level response** gives the mean annual change in relative sea level, in mm per year, at each global grid cell over the specified time period. Positive values indicate rising sea level; negative values indicate falling sea level.
* The **sea-level response uncertainty** gives the uncertainty associated with each relative sea-level response value, also expressed as an average in mm per year over the analysis period.

The **relative sea-level response** (rel) is the sea-level change relative to the solid Earth's surface and is the quantity most directly comparable to the sea level observed at the coast, for example by a tide gauge. It is computed through the absolute sea-level change and the vertical deformation.
The **absolute sea-level response** is the sea-level change as an altimetry sensor (for example on a satellite) would measure.
The **vertical deformation** (solid Earth) describes the vertical motion of the solid Earth that can be associated with the redistribution of land-ice mass. Whenever we remove mass from e.g., continental crust, the load on that crust lightens and the solid Earth will experience slight uplift. 

The relative sea-level response is calculated as

.. math::

   \Delta h_\mathrm{rel} =
   \Delta h_\mathrm{abs} -
   \Delta h_\mathrm{solid\ Earth}

For a detailed description of SELREM and its validation, see: Ludwigsen, C.B., Andersen, O.B., Marzeion, B. et al. Global and regional ocean mass budget closure since 2003. Nat Commun 15, 1416 (2024). https://doi.org/10.1038/s41467-024-45726-w.


Repository Structure
--------------------

The primary functionality is exposed through two entry points:

``sea_level_response.selrem_entrypoint``
    Computes global sea-level response and associated uncertainty from an input mass-balance dataset.

``sea_level_response.slr_ts_entrypoint``
    Extracts sea-level response time series and its uncertainty at specified geographic coordinates from precomputed SLR fields.

Internally, the package uses SELREM to emulate the sea-level response to redistributed land-ice mass.

Data Flow
---------

The Sea-Level Response workflow consists of two processing stages.

1. Compute sea-level response fields from a mass-balance input dataset.
2. Extract sea-level response time series from the resulting gridded products.

And input datasets may originate from:

* Observational mass-balance products.
* Outputs from other DTC Ice Sheets components.
* Ice-sheet projection datasets such as ISMIP6.

The resulting outputs may be visualized in the DTC dashboard or through the prepared Jupyter Notebook workflows.

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

Integration with the DTC Ice Sheets Dashboard
---------------------------------------------

The Sea-Level Response tab of the DTC Ice Sheets dashboard acts as a graphical client for the functionality provided by this repository.

When a user requests a sea-level response calculation through the dashboard, the dashboard invokes the repository's backend workflows using the same processing pipeline described above.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   API <api.rst>
