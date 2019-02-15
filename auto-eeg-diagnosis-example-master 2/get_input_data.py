from auto_diagnosis import run_exp
import logging
import config
import time
import sys
import numpy
import pickle
from dataset import get_all_sorted_file_names_and_labels
from sklearn.externals import joblib
start_time = time.time()

log = logging.getLogger(__name__)
log.setLevel('DEBUG')
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
X,y,y_bin = run_exp(
        config.data_folders,
        config.n_recordings,
        config.sensor_types,
        config.n_chans,
        config.max_recording_mins,
        config.sec_to_cut, config.duration_recording_mins,
        config.test_recording_mins,
        config.max_abs_val,
        config.sampling_freq,
        config.divisor,
        config.test_on_eval,
        config.n_folds, config.i_test_fold,
        config.shuffle,
        config.model_name,
        config.n_start_chans, config.n_chan_factor,
        config.input_time_length, config.final_conv_length,
        config.model_constraint,
        config.init_lr,
        config.batch_size, config.max_epochs,config.cuda,'standard')
end_time = time.time()
run_time = end_time - start_time

log.info("Experiment runtime: {:.2f} sec".format(run_time))



#joblib.dump(X, "X")
joblib.dump(y, "y")
joblib.dump(y_bin, "y_bin")
#joblib.dump(test_X, "test_X")
#joblib.dump(test_y, "test_y")
#joblib.dump(test_y_bin, "test_y_bin")

print('DONE!!!')
