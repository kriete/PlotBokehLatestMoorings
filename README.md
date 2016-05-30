# PlotBokehLatestMoorings
Create bokeh representations of the latest data (incl. QC) from socib mooring stations.

Notes:
We use simple web scraping from the socib thredds server to obtain the opendap links to all latest weather station data.
Also, we create a javascript callback to automatically adjust the y-axis according to the current zoom-extend.

![...](/img/bokeh_latest_data.png?raw=true "HTML bokeh output")
