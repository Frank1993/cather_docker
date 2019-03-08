from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class DBConnection:
    def __init__(self, user, password, host, port, db_name):
        self.engine = create_engine('postgresql://' + user + ':' + password + '@' + host + ':' + port + '/' + db_name,
                                    echo=False)
        self.session = sessionmaker()
        self.session.configure(bind=self.engine)

    def get_session(self):
        return self.session()
