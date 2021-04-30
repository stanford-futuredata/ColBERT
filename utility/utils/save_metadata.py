import os
import sys
import git
import time
import copy
import ujson
import socket


def get_metadata(args):
    args = copy.deepcopy(args)

    args.hostname = socket.gethostname()
    args.git_branch = git.Repo(search_parent_directories=True).active_branch.name
    args.git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    args.git_commit_datetime = str(git.Repo(search_parent_directories=True).head.object.committed_datetime)
    args.current_datetime = time.strftime('%b %d, %Y ; %l:%M%p %Z (%z)')
    args.cmd = ' '.join(sys.argv)

    try:
        args.input_arguments = copy.deepcopy(args.input_arguments.__dict__)
    except:
        args.input_arguments = None

    return dict(args.__dict__)


def format_metadata(metadata):
    assert type(metadata) == dict

    return ujson.dumps(metadata, indent=4)


def save_metadata(path, args):
    assert not os.path.exists(path), path

    with open(path, 'w') as output_metadata:
        data = get_metadata(args)
        output_metadata.write(format_metadata(data) + '\n')

    return data
