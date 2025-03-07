import datetime
import time

start = datetime.datetime.now()

time.sleep(1)
end = datetime.datetime.now() - start
print(end - datetime.timedelta(microseconds=end.microseconds))
