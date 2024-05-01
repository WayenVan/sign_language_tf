
import subprocess

def save_git_diff_to_file(filename):
    try:
        # Run git diff command and redirect the output to the specified file
        with open(filename, 'w') as file:
            subprocess.call(['git', 'diff', 'HEAD'], stdout=file)
        
        print("Git diff output saved to", filename)
    except subprocess.CalledProcessError as e:
        print("Error:", e.output)
    except Exception as e:
        print("An error occurred:", str(e))

def get_current_git_hash():
    try:
        # Run git rev-parse HEAD command to get the hash of the current commit
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        
        # Decode the output bytes to string
        git_hash = output.decode('utf-8')
        return git_hash
    except subprocess.CalledProcessError as e:
        print("Error:", e.output)
        return None
    
def save_git_hash(file):
    hash = get_current_git_hash()
    if hash is not None:
        with open(file, 'w') as file:
            file.writelines([
                '#!/user/bin/bash\n',
                f'git checkout {hash}'
            ])
    else:
        raise RuntimeError("hash is none")
