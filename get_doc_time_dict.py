import json
from os import listdir
from os.path import isfile, join
def is_sgm_file(path):
    if isfile(path) and '.sgm' in path:
        return True
    else:
        return False

subfolder = ['bc','bn','cts','nw','un','wl']
file2time = {}
for sub in subfolder:
    print('loading:',sub,'/adj...')
    mypath = './ace_2005_td_v7/data/English/'+sub+'/adj'
    for f in listdir(mypath):
        if is_sgm_file(join(mypath, f)):
            with open(join(mypath, f)) as fp:
                for i, line in enumerate(fp):
                    if i == 3:
                        assert f not in file2time
                        file2time[f.replace('.sgm','').strip()] = line.replace('<DATETIME>','').replace('</DATETIME>','').strip()
                        break
for sub in subfolder:
    print('loading:',sub,'/timex2norm...')
    mypath = './ace_2005_td_v7/data/English/'+sub+'/timex2norm'
    for f in listdir(mypath):
        if is_sgm_file(join(mypath, f)):
            with open(join(mypath, f)) as fp:
                for i, line in enumerate(fp):
                    if i == 3:
                        if f.replace('.sgm','').strip() not in file2time:
                            file2time[f.replace('.sgm','').strip()] = line.replace('<DATETIME>','').replace('</DATETIME>','').strip()
                            break
# print(file2time)
# print(len(file2time))

doc_set = set()
with open('ACE05_events_three_level_train_with_sent_id.json') as f:
    data = json.load(f)

for item in data:
    doc_set.add(item['DOC_ID'])

with open('ACE05_events_three_level_dev_with_sent_id.json') as f:
    data = json.load(f)
for item in data:
    doc_set.add(item['DOC_ID'])

with open('ACE05_events_three_level_test_with_sent_id.json') as f:
    data = json.load(f)
for item in data:
    doc_set.add(item['DOC_ID'])

print(len(file2time))
print(len(doc_set))
for doc_id in doc_set:
    if doc_id not in file2time:
        print('<'+doc_id+'>') 

# year,month,day,hour,minute,second
def parse_time_shortest(line):
    return {'year':int(line[:4]),'month':int(line[4:6]),'day':int(line[6:8]),'hour':0,'minute':0,'second':0}

def parse_time_T_format_short(line):
    y_m_d = line.split('T')[0].split('-')
    h_m_sec = line.split('T')[1].split(':')
    return {'year':int(y_m_d[0]),'month':int(y_m_d[1]),'day':int(y_m_d[2]),'hour':int(h_m_sec[0]),'minute':int(h_m_sec[1]),'second':int(h_m_sec[2])}


def parse_time_space_format(line):
    y_m_d = line.split(' ')[0].split('-')
    h_m_sec = line.split(' ')[1].split(':')
    return {'year':int(y_m_d[0]),'month':int(y_m_d[1]),'day':int(y_m_d[2]),'hour':int(h_m_sec[0]),'minute':int(h_m_sec[1]),'second':int(h_m_sec[2])}

def parse_time_T_format_long(line):
    y_m_d = line.split('T')[0].split('-')
    h_m_sec = line.split('T')[1].replace('-',':').split(':')
    return {'year':int(y_m_d[0]),'month':int(y_m_d[1]),'day':int(y_m_d[2]),'hour':int(h_m_sec[0]),'minute':int(h_m_sec[1]),'second':int(h_m_sec[2])}

def parse_time_dash_format(line):
    y_m_d = line.split('-')[0].strip()
    h_m_sec = line.split('-')[1].split(':')
    return {'year':int(y_m_d[:4]),'month':int(y_m_d[4:6]),'day':int(y_m_d[6:8]),'hour':int(h_m_sec[0]),'minute':int(h_m_sec[1]),'second':int(h_m_sec[2])}

shotest = {'AFP','APW_ENG','NYT_ENG','XIN_ENG'}
T_format_short = {'AGGRESSIVE','FLOPPINGACES','Austin','BACONSREBELLION','GETTINGPOLITICAL','HEALINGIRAQ','Integritas','MARKBACKER','MARKETVIEW','OIADVANTAGE','TTRACY','alt.','aus.','marcellapr','misc.','rec.','seattle.','soc.','talk.','uk.'}
T_format_long = {'CNN_CF','CNN_IP','CNN_LE'}
space_format = {'CNNHL','CNN_ENG'}
dash_format = {'fsh_'}

for key,value in file2time.items():
    is_parsed = False
    for pfx in shotest:
        if pfx in key:
            file2time[key] = parse_time_shortest(value)
            is_parsed = True
    if is_parsed == False:
        for pfx in T_format_short:
            if pfx in key:
                file2time[key] = parse_time_T_format_short(value)
                is_parsed = True
    if is_parsed == False:
        for pfx in T_format_long:
            if pfx in key:
                file2time[key] = parse_time_T_format_long(value)
                is_parsed = True
    if is_parsed == False:
        for pfx in space_format:
            if pfx in key:
                file2time[key] = parse_time_space_format(value)
                is_parsed = True
    if is_parsed == False:        
        for pfx in dash_format:
            if pfx in key:
                file2time[key] = parse_time_dash_format(value)
                is_parsed = True

with open('doc_id_to_time.json','w') as out_file:
    json.dump(file2time, out_file, indent = 4, sort_keys = True) 
