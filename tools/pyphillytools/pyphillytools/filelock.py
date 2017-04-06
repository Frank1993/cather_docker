import os
import time
import errno
import fcntl
 
class FileLockException(Exception):
    pass
 
class FileLock(object): 
    def __init__(self, file_name, timeout=10, delay=.05):
        self.is_locked = False
        self.lockfile = os.path.join(os.getcwd(), "%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay
 
    # Acqire a file lock
    def acquire(self):
        start_time = time.time()
        while True:
            try:
                self.fd = open(self.lockfile, 'w+')
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.is_locked = True
                break
            except OSError as e:
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException("Timeout occured.")
                time.sleep(self.delay)
 
    # Release the filelock
    def release(self):
        if self.is_locked:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.is_locked = False
 
    # Entering a with statement
    def __enter__(self):
        if not self.is_locked:
            self.acquire()
        return self
 
    # For the exit of a with statement
    def __exit__(self, type, value, traceback):
        if self.is_locked:
            self.release()
 
    # Make sure the lock and file does not remain
    def __del__(self):
        self.release()
        try:
            os.remove(self.lockfile)
        except OSError:
            pass