export PYTHONPATH="${PYTHONPATH}:$(pwd)"

/home/jiajie/anaconda3/envs/TNT/bin/python core/util/preprocessor/argoverse_preprocess_v2.py --root dataset/ --dest dataset

# generate a small subset to test the training program
# /home/jiajie/anaconda3/envs/TNT/bin/python core/util/preprocessor/argoverse_preprocess_v2.py --root dataset/ --dest dataset -s
