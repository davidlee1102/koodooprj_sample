import os.path

import datetime
import pandas as pd

from koodoovoice.model_packages import constant_key


def dataframe_records(request, result, status):
    data_path = constant_key.DATAFRAME_RECORD_PATH
    if os.path.isfile(data_path) is False:
        dict_temp = {
            'time': [datetime.datetime.now()],
            'request': [request],
            'result': [result],
            'status': [status]
        }
        data_frame = pd.DataFrame(dict_temp)
        data_frame.to_csv(data_path, index=False)
    else:
        data_frame = pd.read_csv(data_path)
        data_frame = data_frame.reset_index(drop=True)
        dict_temp = {
            'time': [datetime.datetime.now()],
            'request': [request],
            'result': [result],
            'status': [status]
        }
        data_frame_temp = pd.DataFrame(dict_temp)
        data_frame_temp = data_frame_temp.reset_index(drop=True)
        data_frame = pd.concat([data_frame, data_frame_temp], ignore_index=True)
        data_frame = data_frame.reset_index(drop=True)
        data_frame.to_csv(data_path, mode='w')
