class AbstractMetadataReader():

    SENSOR_NAMES = ['timestamp', 'lon', 'lat', 'elv', 'h_accu', 'GPSs', 'yaw', 'pitch', 'roll',
                    'accX', 'accY', 'accZ', 'pres', 'comp', 'vIndex', 'tFIndex', 'gX', 'gY', 'gZ', 'OBDs', 'v_accu']

    def read_metadata(self, metadata_path):
        pass
    
    def get_device_name(self, metadata_path):
        pass