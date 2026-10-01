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
isostatic adjustment, and redistribution of water due to the Earth's rotation produce characteristic spatial patterns in
sea-level change. Consequently, the sea-level response close to a region of ice loss can differ substantially from 
the global mean response.

This module provides two modes to explore sea-level changes:
* **Current sea-level response** calculates sea-level responses from observational ice-sheet mass-balance data or
  mass-change data produced by other DTC Ice Sheets modules, e.g., the Ice-Sheet Mass-Balance module, or the 
  Ice-Shelves module.
* **Future sea-level response** calculates sea-level response from ice-sheet projections, including model experiments
  from ISMIP6.

For both modes, users can explore the spatial sea-level response on the interactive globe and extract a sea-level 
time series at a specified latitude and longitude.


Scientific background on SLR calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DTC Ice Sheets uses **SELREM (SEa-Level Response EMulator)** to translate spatially distributed changes in
land-ice mass into spatial patterns of sea-level change.

For a detailed description of SELREM and its validation, see :cite:`Ludwigsen, C.B., Andersen, O.B., Marzeion, B. et al. Global and regional ocean mass budget closure since 2003. Nat Commun 15, 1416 (2024). https://doi.org/10.1038/s41467-024-45726-w`.


Current sea-level response
--------------------------

The **Current SLR** view calculates sea-level responses from observed or previously computed ice-sheet mass changes.

Select mass-balance data
~~~~~~~~~~~~~~~~~~~~~~~~

Under **Configure mass balance input data**, select the dataset to use as input. The information box below the 
selection shows a short description of the dataset and the period for which data are available.

Use the **time range** slider to select the analysis period and click **Load mass input data**.

Once the data have loaded, the dashboard reports the mean mass balance and its uncertainty over the selected period.

Calculate the global response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click **Explore global sea-level response** to calculate the spatial sea-level response from the selected 
mass-change data.

The resulting response is displayed on the interactive globe. Use the switches in the legend to enable or disable
layers.

The **Sea level response** layer shows the mean annual change in relative sea levels in mm per year for every location
in the global oceans over the specified time period. Positive values are assiciated with rising sea levels; negative 
values with falling sea levels.
The **Sea level response uncertainty** layer shows the associated uncertainty to each value of relative sea level
response. The uncertainty is also shown in average mm per year over the analysis period.


Explore the response at a location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To examine the sea-level response at a specific location, enter its **latitude** and **longitude** and click
**Plot SLR at location**. The globe will zoom to the selected location and set a pin.

The dashboard now also shows three time series:

**Relative sea-level response** (rel)
    This is the sea-level change relative to the solid Earth's surface and is the quantity most directly comparable to 
    the sea level observed at the coast, for example by a tide gauge. It is also the one computed above.

**Absolute sea-level response** (abs)
    This is the sea-level change as an altimetry sensor (for example on a satellite) would measure.

**Vertical deformation** (solid Earth)
    This describes the vertical motion of the solid Earth that can be associated with the redistribution of land-ice
    mass. Whenever we remove mass from e.g., continental crust, the load on that crust lightens and the solid Earth
    will experience slight uplift. 

The relative sea-level response is calculated as

.. math::
   \Delta h_\mathrm{rel} = \Delta h_\mathrm{abs} - \Delta h_\mathrm{solid\ Earth}.


Future sea-level response
-------------------------

The **Future SLR** view calculates sea-level responses from projected ice-sheet changes simulated in ISMIP6 experiments.


Select an ice-sheet projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select an **ice-sheet model** and an available **experiment**. Information about the selected model run and its
available projection period is shown alongside the controls.

Choose a projection period using the **Time range** slider and click **Load ISMIP6 data**.

The dashboard reports the mean mass balance calculated from the selected model output and period.


Inspect the projected ice-sheet change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click **Visualise cumulative ice thickness change** to display an animation of the changing ice thickness over the 
entire selected period. A timeline at the bottom gives you an overview of the current visualized time point.


Explore the response at a location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here again, you can explore the time series of a specific location. More info, see above.

Quick Start
===========

The package provides two CLI entry points, one to calculate global sea-level response given an input mass balance dataset 
and another to extract a sea-level time series at a specific location.

Run either with ``--help`` to see all options.

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
