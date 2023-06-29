from astropy.io import ascii
from os.path    import join as ospath_join

from ipywidgets \
import  \
Button as wButton, \
Select as wSelect, \
Label  as wLabel, \
Layout as wLayout, \
VBox   as wVBox, \
HBox   as wHBox

from plotly.graph_objs \
import \
Scattergl    as plyScattergl, \
Layout       as plyLayout, \
FigureWidget as plyFigureWidget

from qsotools.fiducial import LYA_WAVELENGTH
from qsotools.io import KODIAQFits

class KODIAQViewer:
    """
    Jupyter notebook widgets to surf, plot and interactively select regions to mask
    for KODIAQ. 
    
    Parameters
    ----------
    kodiaq_dir : str
        Directory of KODIAQ data.
    asu_path : str
        Path to asu.tsv table that contains list of quasars. Obtain from 
        http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/154/114
        table3
    lya_lower : float
        Lya Forest initial wavelength in rest frame in A.
    lya_upper : float
        Lya Forest final wavelength in rest frame in A.
        
    __init__(kodiaq_dir, asu_path, lya_lower, lya_upper)
        Sets up widgets (buttons, selects, labels and figure widget). Reads asu.tsv table.

    Attributes
    ----------
    kodiaq_dir : str
        Directory of KODIAQ data.
    LYA_FIRST : float
        Lya Forest initial wavelength in rest frame in A.
    LYA_LAST : float
        Lya Forest final wavelength in rest frame in A.
    mask_region_list : list
        List of masked regions in A. [[w1i, w1f], [w2i, w2f], ...]
    NUMBER_OF_SUBPLOTS : int
        Number to divide one spectrum. This creates more legible plots.
    w0 : float
        Initial wavelength for a given spectrum.
    dWave : float
        Wavelength interval to show for each section.
    asu_table : astropy.io.ascii
        Stores QSO and properties (such as redshift) from asu.tsv table.
    readme_table : astropy.io.ascii
        Stores observation and spec_prefix from a selected QSO's README.tbl.
    all_qsos : list
        List of QSOs in KODIAQ from asu.tsv.
    plotted_qso : dict
        Dictionary to save plotted QSO, observation, spec_prefix and zqso.
    range_counter : int
        Counter to hold section in view.

    plot_button : ipywidgets.Button
        Plots the selected observation on click.
    backtoinitrange_button : ipywidgets.Button
        Scales back to initial range on click.
    jumprange_button : ipywidgets.Button
        Jumps to the next range in wavelength on click.
    mask_save_button : ipywidgets.Button
        Saves the selected mask regions into observation's folder with QSO_OBS_dlabyeye_mask.dat filename.
    masked_region_remove_button : ipywidgets.Button
        Removes the selected region from mask list.

    pick_a_qso_select :ipywidgets.Select
        Select a QSO.
    pick_an_observation_select :ipywidgets.Select
        Select an observation.
    masked_region_select :ipywidgets.Select
        Select/List for masked regions.
    qso_properties_desc : list of ipywidgets.Label
        Has labels z for redshift of the QSO and DR release for a specific observation.
    qso_properties_items : list of ipywidgets.Label
        Show the z and DR of the QSO.

    fig_wid : plotly.graph_objs.FigureWidget
        Plotting widget. Can select mask regions using Box Select.

    Methods
    -------
    show_all()
        Runs the widget.

    setup_buttons()
    setup_selects()
    setup_labels()
    setup_figure_widget()
        Set up widgets.

    refresh_mask()
        Removes the entire mask list.
    refresh_plot()
        Removes the plotted wave, flux and error.
    set_plot_range(x0, x1, y0, y1)
        Set a range for plot.

    plot_selection_fn(trace, points, selector)
        Saves the interval from Box Select to masked_region_select and mask_region_list.

    on_plot_clicked(b)
        Plots.
    on_initrange_clicked(b)
        Resets to initial plotting range.
    on_jumprange_clicked(b)
        Jumps to next range interval.
    on_qsoselect_change(change)
        Keeps track of selected QSO.
    on_obsselect_change(change)
        Keeps track of selected observation.
    on_removemask_clicked(b)
        Removes the selected range from masked_region_select and mask_region_list.
    on_save_clicked(b)
        Saves the mask_region_list to QSO_OBS_dlabyeye_mask.dat filename.
    
    """
    def setup_buttons(self):
        self.plot_button = wButton(
            description='Plot',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Plot'
        )

        self.backtoinitrange_button = wButton(
            description='Init Range',
            disabled=False,
            button_style='', 
            tooltip='Initial Range'
        )

        self.jumprange_button = wButton(
            description='Jump',
            disabled=False,
            button_style='', 
            tooltip='Jump Range'
        )

        self.mask_save_button = wButton(
            description='Save',
            disabled=False,
            button_style='', 
            tooltip='Save'
        )

        self.masked_region_remove_button = wButton(
            description='Remove Mask',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Remove Mask'
        )

        self.plot_button.on_click(self.on_plot_clicked)
        self.backtoinitrange_button.on_click(self.on_initrange_clicked)
        self.jumprange_button.on_click(self.on_jumprange_clicked)
        self.mask_save_button.on_click(self.on_save_clicked)
        self.masked_region_remove_button.on_click(self.on_removemask_clicked)

    def setup_selects(self):
        self.pick_a_qso_select = wSelect(
            options=self.all_qsos,
            value=self.all_qsos[0],
            rows=10,
            layout=wLayout(width='300px'),
            description='QSO:',
            disabled=False
        )

        readme_fname = ospath_join(self.kodiaq_dir, self.all_qsos[0],"README.tbl")
        self.readme_table = ascii.read(readme_fname)
        obs_all = list(self.readme_table['pi_date'])

        self.pick_an_observation_select = wSelect(
            options=obs_all,
            value= obs_all[0],
            rows=10,
            layout=wLayout(width='300px'),
            description='Obs:',
            disabled=False
        )

        self.masked_region_select = wSelect(
            options=[''],
            value='',
            rows=10,
            layout=wLayout(width='300px'),
            description='Mask:',
            disabled=False
        )

        self.pick_a_qso_select.observe(self.on_qsoselect_change, 'index')
        self.pick_an_observation_select.observe(self.on_obsselect_change, 'index')  

    def setup_labels(self):
        label_str = ['z:', 'DR:']
        label_ly  = wLayout(width='50px')
        self.qso_properties_desc = []
        self.qso_properties_items = []

        for ls in label_str:
            self.qso_properties_desc.append(wLabel(value=ls))
            self.qso_properties_items.append(wLabel(value='', layout=label_ly))

    def setup_figure_widget(self):
        flux_trace = plyScattergl(
            x = [],
            y = [],
            name = 'f',
            line = dict( color = ('rgb(0, 0, 255)')),
            showlegend=True,
            mode = 'lines+markers',
            marker = dict(
                color = '#0000ff',
                size  = 1
            )
        )
        error_trace = plyScattergl(
            x = [],
            y = [],
            name = '1-e',
            line = dict( color = ('rgb(255, 0, 0)')),
            showlegend=True,
            mode = 'lines+markers',
            marker = dict(
                color = '#ff0000',
                size  = 1
            )
        )

        emline_dict={
                    'type': 'line',
                    'x0': 1,
                    'y0': -0.5,
                    'x1': 1,
                    'y1': 1.5,
                    'line': {
                        'color': 'rgb(128, 0, 128)',
                        'width': 5,
                    },
                }

        lyaforest_dict={
                    'type': 'rect',
                    'xref': 'x',
                    'x0': 1,
                    'y0': -0.5,
                    'x1': 1,
                    'y1': 1.5,
                    'fillcolor': '#006400',
                    'opacity': 0.3,
                    'line': {
                        'width': 0,
                    }
                }

        layout = plyLayout({
            'dragmode':'pan',
            'xaxis' : dict(
                range=[2, 5],
                rangeslider=dict(
                    visible = True
                )
            ),
            'yaxis' : dict(
                range=[-0.5, 1.5]
            ),
            'shapes':[emline_dict, lyaforest_dict]
        })

        data = [flux_trace, error_trace]
        self.fig_wid = plyFigureWidget(data=data, layout=layout)
        self.fig_wid['data'][0].on_selection(self.plot_selection_fn)

    def __init__(self, kodiaq_dir, asu_path, lya_lower, lya_upper):
        self.kodiaq_dir = kodiaq_dir

        self.LYA_FIRST = lya_lower
        self.LYA_LAST  = lya_upper

        self.mask_region_list = []
        self.NUMBER_OF_SUBPLOTS = 50

        self.w0    = 0
        self.dWave = 0

        self.plotted_qso = {'QSO' : "", 'PI' : "", 'SP' : "", "ZQSO" : 0}

        # Read table for quasars
        self.asu_table = ascii.read(asu_path, data_start=3)

        self.all_qsos = list(self.asu_table['KODIAQ'])

        # Set up widgets
        self.setup_selects()
        self.setup_buttons()
        self.setup_labels()
        self.setup_figure_widget()

    def refresh_mask(self):
        self.mask_region_list = []
        self.masked_region_select.options = ['']
        self.masked_region_select.value   = ''

    def refresh_plot(self):
        self.range_counter = 0
        self.plotted_qso = {'QSO' : "", 'PI' : "", 'SP' : "", "ZQSO" : 0}

        with self.fig_wid.batch_update():
            self.fig_wid['layout'].update(title='')
            
            self.fig_wid['data'][0].x = []
            self.fig_wid['data'][0].y = []
            self.fig_wid['data'][1].x = []
            self.fig_wid['data'][1].y = []
    
    def set_plot_range(self, x0, x1, y0, y1):
        with self.fig_wid.batch_update():
            layout_fw   = self.fig_wid['layout']

            # Set a smaller range
            layout_fw.xaxis['range'] = [x0, x1]
            layout_fw.yaxis['range'] = [y0, y1]

    def plot_selection_fn(self, trace, points, selector):
        if len(points.point_inds) > 0:
            self.mask_region_list.append([points.xs[0], points.xs[-1]])
            
            if len(self.mask_region_list) == 1:
                self.masked_region_select.options = ("%.3f - %.3f A" % (self.mask_region_list[0][0], self.mask_region_list[0][1]),)
            else:
                self.masked_region_select.options += ("%.3f - %.3f A" % (self.mask_region_list[-1][0], self.mask_region_list[-1][1]),)
                
            self.masked_region_select.value   = self.masked_region_select.options[-1]
    
    def on_plot_clicked(self, b):
        self.refresh_mask()
        self.refresh_plot()

        qso_name = self.pick_a_qso_select.value
        pi_date  = self.pick_an_observation_select.value
        
        index_qso = self.pick_a_qso_select.index
        index_obs = self.pick_an_observation_select.index

        spec_prefix = self.readme_table[index_obs]['spec_prefix']

        if qso_name == "" or pi_date == "":
            return
        
        qso_dir = ospath_join(self.kodiaq_dir, qso_name)
        
        ZQSO    = self.asu_table['zem'][index_qso]
        
        try:
            kod_file = KODIAQFits(self.kodiaq_dir, qso_name, pi_date, spec_prefix, ZQSO)
        except:
            raise
        
        self.plotted_qso = {'QSO' : qso_name, 'PI' : pi_date, 'SP' : spec_prefix, "ZQSO" : ZQSO}

        # kod_file.flux[kod_file.flux < -0.5] = -0.5
        # kod_file.flux[kod_file.flux > 1.5]  = 1.5
        
        # kod_file.error[kod_file.error > 1] = 1
        
        self.w0    = kod_file.wave[0]
        self.dWave = (kod_file.wave[-1] - self.w0) / self.NUMBER_OF_SUBPLOTS

        fig_title = "%s/%s/%s at z=%.2f" % (qso_name, pi_date, spec_prefix, ZQSO)
            
        scatter_arr = self.fig_wid['data']
        layout_fw   = self.fig_wid['layout']

        with self.fig_wid.batch_update():    
            layout_fw.update(title=fig_title)
            # Set a smaller range
            layout_fw.xaxis['range'] = [self.w0, self.w0 + self.dWave]
            
            # Plot the spectrum
            scatter_arr[0].x = kod_file.wave
            scatter_arr[1].x = kod_file.wave

            scatter_arr[0].y = kod_file.flux
            scatter_arr[1].y = 1. - kod_file.error

            # Plot Lya emmission line
            layout_fw['shapes'][0]['x0'] = LYA_WAVELENGTH * (1. + ZQSO)
            layout_fw['shapes'][0]['x1'] = LYA_WAVELENGTH * (1. + ZQSO)
            
            # Shade Lya forest region
            layout_fw['shapes'][1]['x0'] = self.LYA_FIRST * (1. + ZQSO)
            layout_fw['shapes'][1]['x1'] = self.LYA_LAST  * (1. + ZQSO)
            
        del kod_file

    def on_initrange_clicked(self, b):
        self.range_counter = 0

        self.set_plot_range(self.w0, self.w0 + self.dWave, -0.5, 1.5)

    def on_jumprange_clicked(self, b):
        self.range_counter += 1

        if self.range_counter >= self.NUMBER_OF_SUBPLOTS:
            return

        w1 = self.w0 + self.range_counter       * self.dWave
        w2 = self.w0 + (self.range_counter + 1) * self.dWave

        self.set_plot_range(w1, w2, -0.5, 1.5)
            
    def on_qsoselect_change(self, change):
        index = change['new']
        
        if index == None:
            return

        qso_name = self.all_qsos[index]

        readme_fname = ospath_join(self.kodiaq_dir, qso_name,"README.tbl")
        self.readme_table = ascii.read(readme_fname)
    
        self.dr_list = list(self.readme_table['kodrelease'])
        self.pick_an_observation_select.options = self.readme_table['pi_date']
        self.pick_an_observation_select.value = self.readme_table[0]['pi_date']
    
        self.qso_properties_items[0].value = "%.2f" % self.asu_table['zem'][index]
            
    def on_obsselect_change(self, change):
        index = change['new']
        
        if index == None:
            return

        self.qso_properties_items[1].value = str(self.dr_list[index])

    def on_removemask_clicked(self, b):
        if len(self.mask_region_list) == 0:
            return
        
        masked_regions_all = list(self.masked_region_select.options)
        masked_delete      = self.masked_region_select.value
        
        index = masked_regions_all.index(masked_delete)
        
        del masked_regions_all[index]
        del self.mask_region_list[index]
        
        if len(masked_regions_all) == 0:
            index = 0
            masked_regions_all = ['']
        elif index >= len(masked_regions_all):
            index = len(masked_regions_all) - 1
        
        self.masked_region_select.options = masked_regions_all
        self.masked_region_select.value   = masked_regions_all[index]
        
    def on_save_clicked(self, b):
        qso_name = self.plotted_qso['QSO']
        pi_date  = self.plotted_qso['PI']
        spec_prefix = self.plotted_qso['SP']
        
        if qso_name == "" or pi_date == "" or spec_prefix == "" or len(self.mask_region_list) == 0:
            return

        fname_mask = "%s_%s_dlabyeye_mask.dat" %(spec_prefix, pi_date)
        save_path = ospath_join(self.kodiaq_dir, qso_name, pi_date, fname_mask)
        
        toWrite = open(save_path, 'w')
        toWrite.write("%d\n" % len(self.mask_region_list))
        for m in self.mask_region_list:
            temp_s = "%f, %f\n" %(m[0], m[1])
            toWrite.write(temp_s)
            
        toWrite.close()

    def show_all(self):
        self.refresh_mask()
        self.refresh_plot()

        button_box = wVBox([self.mask_save_button, self.masked_region_remove_button])

        qsodesc_box = wVBox(self.qso_properties_desc)
        qsoprop_box = wVBox(self.qso_properties_items)

        desc_plot_box = wVBox([ wHBox([qsodesc_box, qsoprop_box]), \
                                self.plot_button, \
                                self.backtoinitrange_button, \
                                self.jumprange_button])

        qso_mask_box = wHBox([  self.pick_a_qso_select, \
                                self.pick_an_observation_select, \
                                desc_plot_box, \
                                self.masked_region_select, \
                                button_box])

        return wVBox([qso_mask_box, self.fig_wid])



























