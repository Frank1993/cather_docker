from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_orm import *


class DetectionsProvider:
    def __init__(self, username, password, host, database):
        self.__initialize_db_connection(username, password, host, database)
        self.__populate_sign_map()

    def __initialize_db_connection(self, username, password, host, database):
        self.engine = create_engine(
            'mysql+pymysql://' + username + ':' + password + '@' + host + '/' + database)
            #, echo=True) # debug purpose
        session = sessionmaker()
        session.configure(bind=self.engine)
        self.s = session()

    def __populate_sign_map(self):
        sign_types = self.s.query(RoadSignType).all()
        self.sign_map = {sign_type.sign_id: sign_type.sign_type for sign_type in sign_types}

    def get_all_detections_for_trip(self, trip_id):
        trip = self.s.query(Trip).filter(Trip.source_sequence_id == trip_id).first()
        sign_det = self.s.query(TripImageSignDetection, TripImageRoi) \
            .filter(TripImageSignDetection.trip_image_roi_id == TripImageRoi.trip_image_roi_id) \
            .filter(TripImageRoi.trip_id == trip.trip_id) \
            .all()

        return {sign.TripImageRoi.trip_image_index: sign.TripImageSignDetection.road_sign_type.sign_type for sign in
                sign_det}

