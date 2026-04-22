#from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
import subprocess
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()