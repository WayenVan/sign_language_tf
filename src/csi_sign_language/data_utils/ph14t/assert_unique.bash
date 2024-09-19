#!/bin/bash

# Directories to check
dir1="/root/projects/sign_language_transformer/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
dir2="/root/projects/sign_language_transformer/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test"
dir3="/root/projects/sign_language_transformer/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev"

# List all folder names under each directory
folders1=$(ls -d $dir1/*/ | xargs -n 1 basename)
folders2=$(ls -d $dir2/*/ | xargs -n 1 basename)
folders3=$(ls -d $dir3/*/ | xargs -n 1 basename)

# Combine all folder names into one list
all_folders=$(echo -e "$folders1\n$folders2\n$folders3")

# Check for duplicates
duplicates=$(echo "$all_folders" | sort | uniq -d)

if [ -z "$duplicates" ]; then
	echo "All folder names are unique."
else
	echo "Duplicate folder names found:"
	echo "$duplicates"
fi
