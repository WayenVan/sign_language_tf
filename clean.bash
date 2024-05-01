#! /usr/bin/bash
outputs_dir=/home/jingyan/Documents/sign_language_rgb/outputs/train_ddp

file_list=`find $outputs_dir -maxdepth 1`

while IFS= read -r dir; do
    files=`find $dir -type f`
    if [[ ! $files =~ "/checkpoint.pt" ]]; then
        mv $dir "$outputs_dir/removed"
    elif [[ $files =~ "/debug" ]]; then
        mv $dir "$outputs_dir/removed"
    fi
done <<< "$file_list"