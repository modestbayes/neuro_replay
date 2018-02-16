# README #

This repository implements an LeNet (CNN) model in decoding neural activities for a sequence memory task.

### How to get set up? ###

* Download SuperChris_WellTrainedSession.mat in the shared Google Drive (link: https://drive.google.com/open?id=0B_SgbpYpV9TfYnhCcWlzdVFXMGc) and save it as `data/SuperChris_WellTrainedSession.mat`. 

    This .mat file contains a table with each row representing every 0.001 second recording time (50 minutes in total). Below is table column information:

    **Timestamps**: the upper limit of their associated timestamp (bin), i.e. each row contains values associated measurements/events occuring between the previous and current row's timestamp value.
    
	**T?_LFP_Raw**: Raw LFP trace (in voltage)
	
	**T?_LFP_Raw_HilbVals**: Hilbert transformed LFP (phase measurements (range +/- pi))
	
	**T?_LFP_Theta**: The Theta filtered LFP (in voltage)
	
	**T?_LFP_Theta_HilbVals**: Hilbert transformed Theta LFP (phase measurements (range +/- pi))
	
	**T?_LFP_Beta**: Beta filtered LFP (in voltage)
	
	**T?_LFP_Beta_HilbVals**: Hilbert transformed Beta LFP (phase measurements (range +/- pi))
	
	**T?-U?**: binary (0 or 1) indicators of whether there was a spike in that time bin or not for unit? of tetrode?
	
	**Odor?**: (1 or 0) trial odor was presented or not
	
	**Position?**: (1 or 0) position indicator
	
	**InSeqLog**: 1 for InSeq trial, 0 for the rest (nontrials and OutSeq)
	
	**PerformanceLog**: 1(correct), -1(incorrect), 0(null event)
	
	**PokeEvents**: 1(rat initialy enters the port), -1(rat withdraws from the port), 0(null event)
	
	**XvalRatMazePosition/YvalRatMazePosition**: a position value (probably in pixel) or a NaN (position was recorded every 15ms on average, many unrecorded rows)

* Run `to_sql_all_tetrodes_superchris_session1.py`, which writes all the data in `SuperChris_WellTrainedSession.mat` to a SQlite database table.

* Run `rat_data.py`, which extracts data related to trials only from the SQL database. A trial starts with rat's nose-poke, however, we would like to extract data a few seconds before and after a trial, and decode the neural activities during that period. Currently we are extracting -2s to 2s relative to the trial start time (poke time). The relative start and end time can be changed in `data/superchris_session1_cnn.json`: `"test"-->"start_sec"` and `"test"-->"end_sec"`.

* Run `decode_data.py`. It has the following functionalities:
    * Load/extract trials data
    * Train and predict
    * Plot prediction results
    
	You can modify `dd = DecodeData('data/*.json')` at the end of the file in `if __name__ == "__main__":`. If implementing CNN model, use 'data/superchris_session1_cnn.json'. If implementing logistic regression, use 'data/superchris_session1_lr.json'.

### Files ###

* `data/SuperChris_WellTrainedSession.mat`: Raw data. About 3GB. You have to download it from the shared Google Drive. 

* `data/superchris_session1_cnn.json`: This JSON file will be imported to `rat_data.py` and `decode_data.py`. All the parameters in `rat_data.py` and `decode_data.py`should be changed and stored in this file. 

* `to_sql_all_tetrodes_superchris_session1.py`: This file populates raw data from the .mat file to a SQLlite database

* `rat_data.py`: This file extracts data associated with trials only. The data is saved as a .pkl file.

* `decode_data.py`: This file implements supervised classification for sequence memory replay neural decoding.

* `trial.py`: Define class Trial

* `config.py`: Configure all the parameters in a JSON file to a python dictionary

* `lr_trainer.py`: Logistic regression trainer

* `cnn_trainer.py`: CNN (LeNet) trainer

* `mat_col_desc.txt`: Table (.mat file) column description# spike_train
# spike_train
