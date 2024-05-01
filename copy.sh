#! /usr/bin/sh

original=/home/jingyan/Documents/sign_language_rgb
target=/home/jingyan/Documents/copy_from_rgb

new_name=sign_language_flownet
folder_name=$(basename $original)

cd $original 
excluded="outputs preprocessed resources dataset"
command="rsync -av $original $target"

for item in $excluded; do 
    command="$command --exclude='$item/'"
done
eval $command

cd $target
mv ./$folder_name ./$new_name
cd $new_name

for item in $excluded; do 
    ln -s $original/$item ./$item
done

mv ./sign_language_rgb.code-workspace ./$new_name.code-workspace
rm ./copy.sh
rm -rf .git
