import time


def sub_millisecond_sleep(threshold):
    start = time.time()
    while True:
        offset = time.time() - start
        if offset > threshold:
            break
