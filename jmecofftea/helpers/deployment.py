#!/usr/bin/env python

import tarfile
import os

from jmecofftea.helpers.paths import jmecofftea_path


def get_repo_files():
    '''Returns a list of tracked files in the jme-cofftea repo'''
    import git

    repo = git.Repo(jmecofftea_path('..'))

    to_iterate = [repo.tree()]
    to_add = []

    while len(to_iterate):
        for item in to_iterate.pop():
            if item.type == 'tree':
                to_iterate.append(item)
            elif item.type == 'blob':
                to_add.append(item.abspath)
    return to_add



def pack_repo(path_to_gridpack):
    '''Creates a gridpack containing the jme-cofftea repo'''
    if os.path.exists(path_to_gridpack):
        raise RuntimeError(f"Gridpack file already exists. Will not overwrite {path_to_gridpack}.")
    tar = tarfile.open(path_to_gridpack,'w')
    files = get_repo_files()
    for f in files:
        tar.add(
            name=f,
            arcname=f.replace(os.path.abspath(jmecofftea_path('..')),'jme-cofftea'),
            exclude=lambda x: ('tgz' in x or 'submission' in x)
            )
    tar.close()
    return