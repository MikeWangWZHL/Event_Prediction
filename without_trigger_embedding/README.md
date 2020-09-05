# Todo 
Hi Emma, the main file you may want to look into is
> prepare_input_ACE05_with_sent_id.py

the main function is `prepare_input_withIBO`,
the three dicts as input **event_type_dict, entity_type_dict, role_type_dictare** are for additional embedding. 
I have added some comments for better understanding.
At line 319, I added what I think may be the best way to extend this for multi events.
Please feel free to let me know if you have any question

Other files related are: 
## ACE05 oneie file: (raw data for training):
> train.oneie.json

> dev.oneie.json

> test.oneie.json

## three level temp file: (extracted all lines with events, and create temps)
> ACE05_events_three_level_train_with_sent_id.json

> ACE05_events_three_level_dev_with_sent_id.json 

> ACE05_events_three_level_test_with_sent_id.json 

## train script
> train_ACE05.py

## Emma's data in json file:  
> all_files_updata.json
