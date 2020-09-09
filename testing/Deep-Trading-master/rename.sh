for d in */ ; do
   if [ "${FILE##*.}" = "ipynb" ]; then
      ipython nbconvert --to=python d
   fi
done