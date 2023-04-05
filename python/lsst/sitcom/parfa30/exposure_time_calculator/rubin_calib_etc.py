import os
import yaml
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interpolate
from scipy.integrate import simps
import matplotlib.pyplot as plt

TPUT_DIR = './baseline_tput_curves'

class RubinCalibETC(object):

    def __init__(self, configuration_file):

        # self.light_source = light_source #LED, laser
        # self.calib_type = calib_type #CBP, Flatfield
        # self.snr = snr

        self.rubin_wavelength = np.linspace(320, 1125, (1125 - 320)+1)

        with open(configuration_file,'r') as f:
            conf = yaml.safe_load(f)
            self.__dict__.update(conf)

        #white light source definition
        

        self.screen_reflectance_files = ['labsphere_outer_ring_reflectance.csv',
                                        'labsphere_inner_ring_reflectance.csv']

        #Telescope&Camera definition
        self.mirror_files = {'Al-Ideal':'m1_ProtAl_Ideal.dat','Al-Aged':'m1_ProtAl_Aged.dat',
                       'Al-Ag':'protected_Al-Ag.csv'}
        # self.f_lsst = f_lsst
        # self.pixel_size = pixel_size
        self.filters = {'u':[324,395.0],'g':[405,552,],'r':[552,691] ,'i':[691,818],
                        'z':[818,921] ,'y4':[922,997]} #,None:[320, 1125]
        self.filter_transmission = {}


        self.photons_per_pixel = {}

        
    def load_led_data(self):
        led_files = ['Mounted_LED_Spectra_MxxxLx-Scaled_IR_E3_2.csv',
                     'Mounted_LED_Spectra_MxxxLx-Scaled_UV_E1.csv',
                     'Mounted_LED_Spectra_MxxxLx-Scaled_Visible_E1.csv']

        dfs = []
        for filen in led_files:
            names = []
            df = pd.read_csv(os.path.join(TPUT_DIR, filen), header=0, skiprows=[1,2,3])
            for i, name in enumerate(list(df.columns)):
                if '#' in name:
                    names.append(None)
                elif 'M' in name:
                    names.append('{}_wave'.format(name))
                    names.append('{}_flux'.format(name))
                    names.append(None)

            df.columns = names[0:-1]
            del df[None]
            dfs.append(df)

        led_df = pd.concat(dfs)
        return led_df

    def led_output(self):
        led_df = self.load_led_data()

        #get dichroic data
        dichroic_data = {}
        for filter_name, leds in self.LEDS.items():
            try:
                dich = pd.read_csv(os.path.join(TPUT_DIR, self.dichroic_files[filter_name]), usecols=[2,3,4], skiprows=0, header=1, names=['wave','trans','refl'])
                trans = interpolate.interp1d(dich.wave, dich.trans/100., fill_value='extrapolate')
                refl = interpolate.interp1d(dich.wave, dich.refl/100., fill_value='extrapolate')
                dichroic_data[filter_name] = [refl, trans]
            except:
                dichroic_data[filter_name] = [1,1]
    
        #multiply LED outputs by the dichroics
        self.led_flux = {}
        for filter_name, leds in self.LEDS.items():
            if filter_name in ['u', 'y4']:
                led = leds[0]
                g = interpolate.interp1d(led_df['{}_wave'.format(led)], led_df['{}_flux'.format(led)], 
                                         fill_value='extrapolate')
                self.led_flux[filter_name] = g(self.rubin_wavelength)/1000. #mW to W
            else:
                total_led_flux = []
                for i, led in enumerate(leds):
                    f = interpolate.interp1d(led_df['{}_wave'.format(led)], led_df['{}_flux'.format(led)]*dichroic_data[filter_name][i](led_df['{}_wave'.format(led)]), 
                                             fill_value='extrapolate')
                    total_led_flux.append(f(self.rubin_wavelength))
                self.led_flux[filter_name] = np.sum(np.vstack(total_led_flux), axis=0)/1000. #mW to W

    def nt242_output(self):
        """The current file used (PGD151_NT242.txt) is there best expected.
        This will be updated to be as expected.
        1000 pulses per second
        decrease_expected: None if measured, if expect actual power to be decrease by this amount (in W)
        """
        wave_m = self.rubin_wavelength * 1e-9 #nm to m
        laser_df = pd.read_csv(os.path.join(TPUT_DIR,self.laser_power_file),delim_whitespace=True)
        laser_energy_output = scipy.interpolate.interp1d(laser_df['Wavelength'],laser_df['Avg'],bounds_error=False, fill_value='extrapolate')
        self.laser_flux = laser_energy_output(self.rubin_wavelength) * 1e-6 * 1000 # uJ to W
        
        if self.laser_decrease_expected is not None:
             self.laser_flux *=  (1-self.laser_decrease_expected)

    def fiber_attenuation(self):
        # data from https://www.ceramoptec.com/products/fibers/optran-uv-/-wf.html
        # convert (dB/km) to transmission
        
        fiber_att_data = pd.read_csv(os.path.join(TPUT_DIR,'ceramoptic_fiber_attenuation.csv'), skiprows=1)
        fiber_att_db_per_km = scipy.interpolate.interp1d(fiber_att_data['Wavelength'], fiber_att_data[self.fiber_type],bounds_error=False, fill_value='extrapolate')
        dB = fiber_att_db_per_km(self.rubin_wavelength) * (self.fiber_length/1000.)
        self.fiber_transmission = (10**(-dB/10.0)) * self.fiber_coupling


    def get_flux(self):
        if self.light_source == 'LED':
            self.led_output()
            return self.led_flux
        elif self.light_source == 'laser':
            self.nt242_output()
            if self.use_fiber:
                self.fiber_attenuation()
                self.laser_flux *= self.fiber_transmission
            return self.laser_flux
        else:
            return('Not a valid light source [LED, laser]')

    def get_photon_rate(self, watts, wavelength):
        #Watts to photons/sec
        h = 6.626e-34 # m2 kg / s
        c = 3e8 # m/s
        wave_m = wavelength * 1e-9 #nm to m
        photon_rate = watts/((h*c)/wave_m)  # ph/s
        return photon_rate

    def get_exptime(self, ph_rate, snr):
        exptime = (snr**2)/ph_rate
        return exptime

    def get_snr(self, ph_rate, exptime):
        snr = np.sqrt(exptime*ph_rate) 
        return snr

    def get_calibration_system_optics_tput(self):
        """Projector efficiency comes from Zemax, different for each system
        Reflector reflectance performed by Patrick Dunlop with a Minolta reflectometer on March 3, 2023
        According to the documentation from Labsphere, we have the results from the reflectance of the screen 

        """
        
    #     # Remove filter dependence in model at 660 nm. for the projector efficiency potentially
    #     r_filter = pd.read_csv(os.path.join(tput_dir, 'ideal_r.dat'),skiprows=4, delim_whitespace=True, names=['wave','transmission'])
    #     r_filter_660 = float(r_filter[r_filter.wave == 660.]['transmission'])
    #     filterless_tput = total_tput/r_filter_660
        self.get_screen_reflectance()
        self.get_projector_optics_throughput()

        self.total_flatfield_optics_tput = {}
        for filter_name in self.filters.keys():
            self.total_flatfield_optics_tput[filter_name] = 1
            self.total_flatfield_optics_tput[filter_name] *= self.projector_mask_efficiency #Coupling from fiber to output of projector
            self.total_flatfield_optics_tput[filter_name] *= self.projector_optics_throughput[filter_name]
            self.total_flatfield_optics_tput[filter_name] *= self.system_efficiency #Coupling between projector and reflector
            self.total_flatfield_optics_tput[filter_name] *= self.reflector_reflectance #Reflective properties of aluminum. This changes over the surface, so this is an average value
            self.total_flatfield_optics_tput[filter_name] *= self.screen_reflectance #Reflective properties of LabSphere
            self.total_flatfield_optics_tput[filter_name] *= self.telescope_acceptance_ratio #Reflective properties of LabSphere

    def get_projector_optics_throughput(self):

        total_tput = {}
        for filter_name in self.filters.keys():
            total_tput[filter_name] = []
            for item in self.projector_design:
                if item == 'collimator':
                    if self.light_source == 'LED':
                        coating = self.coating_type[filter_name]
                        df = pd.read_csv(os.path.join(TPUT_DIR, f'{coating}_Broadband_AR-Coating.csv'), skiprows=2,usecols=[0,1],names=['Wavelength','Reflectance'])
                        f = scipy.interpolate.interp1d(df.Wavelength, 1 - df.Reflectance/100., bounds_error=False, fill_value='extrapolate')
                    elif self.light_source == 'laser':
                        df = pd.read_csv(os.path.join(TPUT_DIR, 'Uncoated_B270_Transmission.csv'), skiprows=2,usecols=[0,1],names=['Wavelength','Transmission'])
                        f = scipy.interpolate.interp1d(df.Wavelength, df.Transmission/100., bounds_error=False, fill_value='extrapolate')
                    total_tput[filter_name].append(f(self.rubin_wavelength))

                elif 'converging_' in item:
                    if item == 'converging_ab':
                        df = pd.read_csv(os.path.join(TPUT_DIR, 'AB_Broadband_AR-Coating.csv'), skiprows=2,usecols=[0,1],names=['Wavelength','Reflectance'])
                        f = scipy.interpolate.interp1d(df.Wavelength, 1 - df.Reflectance/100., bounds_error=False, fill_value='extrapolate')
                    elif item == 'converging_uncoated':
                        df = pd.read_csv(os.path.join(TPUT_DIR, 'Uncoated_N-BK7_Transmission.csv'), skiprows=2,usecols=[0,1],names=['Wavelength','Transmission'])
                        f = scipy.interpolate.interp1d(df.Wavelength, df.Transmission/100., bounds_error=False, fill_value='extrapolate')
                    total_tput[filter_name].append(f(self.rubin_wavelength))

                elif 'mirror' in item:
                    if 'al' in item:
                        df = pd.read_csv(os.path.join(TPUT_DIR,'Aluminum_Coating_Comparison_Data.csv'),
                     usecols=[2,3,4,5], skiprows=[0,1], 
                        header=2, names=['wave_f01','refl_f01','wave_g01','refl_g01'])
                        if 'f01' in item:
                            f = scipy.interpolate.interp1d(df.wave_f01*100, df.refl_f01/100., bounds_error=False, fill_value='extrapolate')
                        elif 'g01' in item:
                            f = scipy.interpolate.interp1d(df.wave_g01*100, df.refl_g01/100., bounds_error=False, fill_value='extrapolate')
                    if 'ag' in item:
                        df = pd.read_csv(os.path.join(TPUT_DIR,'Silver_Coating_Comparsion_Data.csv'),
                                                    usecols=[2,3,4,5], skiprows=[0,1], 
                                                    header=2, names=['wave_p01','refl_p01','wave_p02','refl_p02'])
                        if 'p01' in item:
                            f = scipy.interpolate.interp1d(df.wave_p01*100, df.refl_p01/100., bounds_error=False, fill_value='extrapolate')
                        elif 'p02' in item:
                            f = scipy.interpolate.interp1d(df.wave_p02*100, df.refl_p02/100., bounds_error=False, fill_value='extrapolate')
                    total_tput[filter_name].append(f(self.rubin_wavelength))

        self.projector_optics_throughput = {}
        for filter_name in self.filters.keys():
            self.projector_optics_throughput[filter_name] = np.product(total_tput[filter_name], axis=0)
        




    def get_screen_reflectance(self):
        """
        Screen reflectance. From measured values from LabSphere on Docushare. 
        https://docushare.lsst.org/docushare/dsweb/View/Collection-10467
        Took random file from outer ring measurement REPORT NUMBER:109153-1-10 and inner ring REPORT NUMBER:109153-1-22. Take mean of them
        """
        dfs = [pd.read_csv(os.path.join(TPUT_DIR, filen)) for filen in self.screen_reflectance_files]

        refl = []
        for df in dfs:
            f = scipy.interpolate.interp1d(df.Wavelength, df.Reflectance/100.,bounds_error=False,fill_value='extrapolate')
            refl.append(f(self.rubin_wavelength))

        self.screen_reflectance = np.mean(refl, axis=0)

    def get_integrating_sphere_output(self):
        # Assume 3P-GPS-060-SF AS-02266-060 
        # Spectraflect, 6 inch diameter sphere with entrance pupil 1in, 1in output, and a 2.5 exit pupil
        # All calculations taken from here: https://www.labsphere.com/wp-content/uploads/2021/09/Integrating-Sphere-Theory-and-Applications.pdf
        # Port and sphere diameters are all in inches
        # rho is the mean reflectance of spectralon
        
        sphere_radius = (self.sphere_diameter/2)*0.0254 #m
        sphere_area = 4.0 * np.pi * (sphere_radius**2) # m
        
        port_area = 0
        for port_diameter in self.port_diameters:
            port_radius = (port_diameter/2.) * 0.0254 #m
            port_area += np.pi * port_radius**2
            
        f = port_area/sphere_area #port fraction

        self.Ls = (1 / (np.pi * sphere_area)) * (self.sphere_reflectance/(1-self.sphere_reflectance*(1-f))) # eq 12 - photons/m2, expecting this to be multiplied by the irradiance
        
    def get_mask_efficiency(self):
        """
        exit_port_diameter in inches
        distance of mask in in
        pinhole size in um
        
        First measure SA from output of integrating sphere to mask, then multiply by size of the actual mask
        """
        exit_port_radius = (self.exit_port_diameter/2.) * 0.0254 #m
        distance_to_mask_m = self.distance_to_mask * 0.0254 #m
        theta = np.arctan(exit_port_radius/distance_to_mask_m)
        SA = np.pi * np.sin(theta)**2 # eqn. 23
        
        pinhole_radius = (self.pinhole_size) / 2. #m

        area_of_mask = np.pi * pinhole_radius**2
        self.mask_efficiency = area_of_mask * SA #eqn 22
        
    def get_cbp_efficiency(self):
        """
        First measure the efficiency of the optical system, then multiply by the transmission of the optics
        """
        light_gathering_power = np.pi / (2*self.f_num_cbp)**2
        self.cbp_efficiency = light_gathering_power * self.cbp_transmission

        self.mag = self.f_lsst/self.f_cbp    
        spot_diam_pixels = (self.pinhole_size * self.mag) / self.pixel_size
        self.spot_total_pixels = np.pi * (spot_diam_pixels/2.)**2

    def get_cbp_throughput(self):
        self.get_integrating_sphere_output()
        self.get_mask_efficiency()
        self.get_cbp_efficiency()
        self.cbp_system_throughput =  self.Ls * self.mask_efficiency * self.cbp_efficiency

    def get_telescope_reflectance(self):
        """
        options: Al-Ideal, Al-Aged, Al-Ag
        Note, the files for each mirror are identical
        projected Al-Ag https://docushare.lsst.org/docushare/dsweb/View/Collection-1047
        All Al reflectances come from https://docushare.lsst.org/docushare/dsweb/View/Collection-1777
        """
        total_reflectance = []
        for mirror in [self.m1, self.m2, self.m3]:
            if mirror == 'Al-Ag':
                df = pd.read_csv(os.path.join(TPUT_DIR, self.mirror_files[mirror]))
                df['Reflectance'] = df.Reflectance/100.
                
            else:
                df = pd.read_csv(os.path.join(TPUT_DIR, self.mirror_files[mirror]),
                                delim_whitespace=True, skiprows=2,names=['Wavelength','Reflectance'])
                
            mirror_tput = scipy.interpolate.interp1d(df.Wavelength, df.Reflectance, 
                                                    bounds_error=False, fill_value='extrapolate')
            refl = mirror_tput(self.rubin_wavelength)
            total_reflectance.append(refl)
            
        self.telescope_reflectance = np.prod(total_reflectance,axis=0) 

    def get_filter_response(self):
        """Filter names: ugriz, y4 or None
        Filter bandpasses taken from Collection-1777
        """

        for filter_name in self.filters.keys():
            if filter_name == None:
                self.filter_transmission[filter_name] = np.ones(len(self.rubin_wavelength))
            else:
                filter_trans = pd.read_csv(os.path.join(TPUT_DIR,'ideal_{}.dat'.format(filter_name)), 
                                        delim_whitespace=True, skiprows=2,names=['Wavelength','Throughput'])

                # Smooth to find edges of filter - No longer doing this. Changes exp time by 0.1 seconds.
                min_trans = 0.01
                smoothed_filter_profile = np.convolve(filter_trans.Throughput, [0,1,1,1,0], mode='same')
                idx = np.where((smoothed_filter_profile > min_trans)&(filter_trans.Wavelength < self.filters[filter_name][1]+50))
                filter_bandpass = filter_trans.iloc[idx[0]]

                filter_trans = scipy.interpolate.interp1d(filter_bandpass.Wavelength, filter_bandpass.Throughput, bounds_error=False, fill_value=0)
                self.filter_transmission[filter_name] = filter_trans(self.rubin_wavelength)


    def get_detector_efficiency(self):
        """Currently only have the e2v efficiency. Should confirm this is up to date and add some others
        """
        
        det_filename = os.path.join(TPUT_DIR, self.detector_file)
        det_tput = pd.read_csv(det_filename, delim_whitespace=True, skiprows=5, names=['Wavelength','Throughput']) 
        
        det_eff = scipy.interpolate.interp1d(det_tput['Wavelength'], det_tput['Throughput'],bounds_error=False, fill_value='extrapolate')
        self.detector_efficiency = det_eff(self.rubin_wavelength)

    def get_telescope_camera_tput(self):
        self.get_telescope_reflectance()
        self.get_filter_response()
        self.get_detector_efficiency()
        self.tel_cam_system_tput = {}
        for filter_name in self.filters.keys():
            self.tel_cam_system_tput[filter_name] = self.telescope_reflectance * self.filter_transmission[filter_name] * self.detector_efficiency
    
    def add_overheads(self, exptime):
        final_exptime = 0
        if exptime <= self.min_exptime:
            final_exptime = self.min_exptime
            num_exposures = 1
        else:
            num_exposures = np.ceil(exptime/self.min_exptime)
            final_exptime = num_exposures * self.min_exptime
        
        readout_time = 0
        for readout in [self.cam_readout, self.electrometer_readout, self.spectrograph_readout]:
            if readout > 0:
                if readout > readout_time:
                    readout_time = readout

        final_exptime += readout_time*num_exposures
        return final_exptime

    def get_photons_per_pixel(self):
        watts_from_lightsource = self.get_flux()
        if self.calib_type == 'CBP':
            self.get_cbp_throughput()
            self.get_telescope_camera_tput()
            if self.light_source == 'laser':
                photons_on_telescope = self.get_photon_rate(watts_from_lightsource * self.cbp_system_throughput, self.rubin_wavelength)
            elif self.light_source == 'LED':
                for filter_name in self.filters.keys():
                    photons_on_telescope = self.get_photon_rate(watts_from_lightsource[filter_name] * self.cbp_system_throughput, self.rubin_wavelength)
            for filter_name in self.filters.keys():        
                photons_detected = photons_on_telescope * self.tel_cam_system_tput[filter_name]
                self.photons_per_pixel[filter_name] = photons_detected / self.total_number_of_pixels
        
        elif self.calib_type == 'Flatfield':
            self.get_calibration_system_optics_tput()
            self.get_telescope_camera_tput()
            if self.light_source == 'laser':      
                for filter_name in self.filters.keys():
                    photons_on_telescope = self.get_photon_rate(watts_from_lightsource * self.total_flatfield_optics_tput[filter_name], self.rubin_wavelength)
                    photons_detected = photons_on_telescope * self.tel_cam_system_tput[filter_name]
                    self.photons_per_pixel[filter_name] = photons_detected / self.total_number_of_pixels
            elif self.light_source == 'LED':
                for filter_name, (filter_start, filter_end) in self.filters.items():
                    center_wave = (filter_end - filter_start)/2. + filter_start
                    photons_on_telescope = self.get_photon_rate(watts_from_lightsource[filter_name] * self.total_flatfield_optics_tput[filter_name], center_wave)

                    photons_detected = photons_on_telescope * self.tel_cam_system_tput[filter_name]
                    self.photons_per_pixel[filter_name] = photons_detected / self.total_number_of_pixels
        

    def get_total_exptime(self):
        self.get_photons_per_pixel()
        exptimes = {}
        if self.calib_type == 'CBP':
            
            for filter_name, (wave_start, wave_end) in self.filters.items():
                spot_flux = self.photons_per_pixel[filter_name] * self.spot_total_pixels
                spot_flux_interp = scipy.interpolate.interp1d(self.rubin_wavelength, spot_flux, 
                                                       bounds_error=False, fill_value='extrapolate')
                
                exptimes[filter_name] = []
                for wave in np.linspace(int(wave_start), int(wave_end), (int(wave_end)-int(wave_start)) + 1):
                    exp_time = self.get_exptime(spot_flux_interp(wave), self.snr)
                    exptimes[filter_name].append((wave, self.add_overheads(exp_time)))
            

        if self.calib_type == 'Flatfield':
            for filter_name in self.filters.keys():
                if self.light_source == 'LED':
                    ph = self.photons_per_pixel[filter_name]
                    integrated_ph_rate = simps(ph[~np.isnan(ph)], self.rubin_wavelength[~np.isnan(ph)])
                    exp_time = self.get_exptime(integrated_ph_rate, self.snr)
                    exptimes[filter_name] = (integrated_ph_rate, self.add_overheads(exp_time))
                elif self.light_source == 'laser':
                    for filter_name, (wave_start, wave_end) in self.filters.items():
                        exptimes[filter_name] = []
                        photon_flux_interp = scipy.interpolate.interp1d(self.rubin_wavelength, self.photons_per_pixel[filter_name])
                        for wave in np.linspace(int(wave_start), int(wave_end), (int(wave_end)-int(wave_start)) + 1):
                            exp_time = self.get_exptime(photon_flux_interp(wave), self.snr)
                            exptimes[filter_name].append((wave, self.add_overheads(exp_time)))
        
        self.total_exptimes = {}
        for filter_name in self.filters.keys():
            exptimes_ = np.array(exptimes[filter_name])
            if self.light_source == 'LED':
                self.total_exptimes[filter_name] = exptimes_[1]
            else:
                self.total_exptimes[filter_name] = np.sum(exptimes_[:,1])

        return exptimes

    def get_integrated_snr(self):
        self.get_photons_per_pixel()
        exptimes = self.get_total_exptime()
        total_snr = {}
        for filter_name in self.filters.keys():
            exptimes_ = np.array(exptimes[filter_name])
            ph = scipy.interpolate.interp1d(self.rubin_wavelength, self.photons_per_pixel[filter_name])
            snr = self.get_snr(ph(exptimes_[:,0]), exptimes_[:,1])
            total_snr[filter_name] = np.sum(snr)
        return total_snr

    def plot_projector_transmission(self):

        self.get_projector_optics_throughput()
        plt.figure(figsize=(8,4))
        for filter_name in self.filters.keys():
            plt.plot(self.rubin_wavelength, self.projector_optics_throughput[filter_name],label=filter_name)
        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Transmission')
        plt.title('Projector Transmission')


    def plot_exptime(self):
        exptimes = self.get_total_exptime()
        plt.figure(figsize=(6,3))
        if self.light_source == 'laser':
            for filter_name, exptime_data in exptimes.items():
                exptime_data = np.array(exptime_data)
                plt.plot(exptime_data[:,0], exptime_data[:,1], label=f'{filter_name}: {np.sum(exptime_data[:,1])/60.:.2f} min.')
            plt.xlabel('Wavelength (nm)')
        if self.calib_type == 'CBP':
            plt.ylabel('Exptime per spot [s]')
        elif self.calib_type == 'Flatfield':
            plt.ylabel('Exptime wavelength [s]')
        plt.title(f'Exposure time for SNR={self.snr} with {self.light_source} on {self.calib_type}')
        
        plt.legend()

    def plot_photon_rate(self):
        plt.figure(figsize=(8,4))
        if self.calib_type == 'CBP':
            for filter_name in self.filters.keys():
                spot_flux = spot_flux = self.photons_per_pixel[filter_name] * self.spot_total_pixels
                spot_flux_ = scipy.interpolate.interp1d(self.rubin_wavelength, spot_flux, 
                                                       bounds_error=False, fill_value='extrapolate')
                plt.plot(self.rubin_wavelength, spot_flux_(self.rubin_wavelength), label=filter_name)
                plt.ylabel('Photons per Spot')
        else:
            self.get_photons_per_pixel()
            for filter_name, photons in self.photons_per_pixel.items():
                plt.plot(self.rubin_wavelength, photons, label=filter_name)
                plt.ylabel('Photons per Pixel')

        plt.legend()
        plt.xlabel('Wavlength (nm)')
        plt.title(f'Photon Rate from {self.light_source} for {self.calib_type}')



    def plot_lightsource_output(self, fiber_plot=False):
        self.get_flux()   
        plt.figure(figsize=(8,4))
        if self.light_source == 'LED':
            for filter_name, flux in self.led_flux.items():
                plt.plot(self.rubin_wavelength, flux, label=filter_name)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Flux [W]')
            plt.title('White Light Source Flux')
            plt.legend()

        elif self.light_source == 'laser':
            plt.plot(self.rubin_wavelength, self.laser_flux)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Flux [W]')
            if self.use_fiber == True:
                plt.title(f'Laser Output: {self.laser_power_file} with Fiber Attenuation')
            else:
                plt.title(f'Laser Output: {self.laser_power_file}')

        if fiber_plot:
            self.fiber_attenuation()
            plt.figure(figsize=(8,4))
            plt.plot(self.rubin_wavelength, self.fiber_transmission,
                     label='Fiber length: {} m\nCoupling: {}'.format(self.fiber_length, self.fiber_coupling))
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Fiber Transmission")
            plt.legend()


    def plot_cbp_throughput(self):
        self.get_cbp_throughput()
        light_flux = self.get_flux()

        plt.figure(figsize=(6,3))
        plt.plot(self.rubin_wavelength, self.cbp_system_throughput * light_flux)
        plt.ylabel('CBP Output Irradiance [W/m2]')
        plt.xlabel('Wavelength (nm)')


    def plot_telescope_and_camera(self, telescope=True, filters=True, detector=True):      
        if telescope:
            self.get_telescope_reflectance()
            plt.figure(figsize=(6,3))
            plt.plot(self.rubin_wavelength, self.telescope_reflectance)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title(f'Telescope; m1:{m1}, m2:{m2}, m3:{m3}')
        if filters:
            self.get_filter_response()
            plt.figure(figsize=(6,3))
            for filter_name in self.filters.keys():
                plt.plot(self.rubin_wavelength, self.filter_transmission[filter_name], label=filter_name)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel('Transmission')
            plt.title('Rubin Filter Transmission')
            plt.legend()

        if detector:
            self.get_detector_efficiency()
            plt.figure(figsize=(6,3))
            plt.plot(self.rubin_wavelength, self.detector_efficiency)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel('QE')
            plt.title(f'Detector Throughput: {self.detector_file}')
        


# if __name__ == '__main__':
#     WL_Flat = RubinCalibETC('LED', 'Flatfield', 100, 'calib_etc.yaml')
#     #WL_Flat.plot_exptime()
#     WL_Flat.plot_photon_rate(spot=False)
#     WL_Flat.get_total_exptime()
#     print(WL_Flat.total_exptimes)
    
#     Laser_Flat = RubinCalibETC('laser', 'Flatfield',100, 'calib_etc.yaml')
#     Laser_Flat.plot_exptime()
#     print(Laser_Flat.total_exptimes)
#     Laser_Flat.plot_photon_rate(spot=False)
    
#     CBP = RubinCalibETC('laser', 'CBP',100, 'calib_etc.yaml')
#     CBP.plot_exptime()
#     print(CBP.total_exptimes)
#     CBP.plot_photon_rate(spot=False)
#     plt.show()
#     Laser = RubinCalibETC('laser')
#     Laser.plot_photon_rate(spot=False)
#     #print(Laser.spot_total_pixels)
#     #Laser.plot_exptime_per_spot()


