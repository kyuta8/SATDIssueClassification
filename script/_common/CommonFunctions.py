import os
import json



def create_dir(path:str):
    _dirs = path.split('/')
    _path = _dirs[0]
    _epoch = 1
    _max_count = len(_dirs)
    while _epoch < _max_count:
        _path = os.path.join(_path, _dirs[_epoch])
        if not(os.path.exists(_path)):
            os.mkdir(_path)
        _epoch += 1


def read_json(path:str):
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


def write_json(path:str, data:dict):
    with open(path, mode='w') as f:
        json.dump(data, f, indent=4)


def read_issues(project: str):
    path = './../data/issue/{}'.format(project)
    read_list = os.listdir(path)
    _issues = {}
    for f in read_list:
        if '.json' in f:
            _issues.update(read_json(os.path.join(path, f)))
            
    index = 0
    issues = {}
    for i in _issues:
        if (_issues[i]['state'] == 'closed') and not(_issues[i].get('pull_request')):
            issues.update({index: _issues[i]})
            index += 1
    return issues