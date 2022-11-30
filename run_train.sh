#
#
#
mkdir results

python3 train.py --input data/Books_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Clothing_Shoes_and_Jewelry_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Home_and_Kitchen_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Electronics_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Movies_and_TV_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Kindle_Store_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Automotive_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

python3 train.py --input data/Grocery_and_Gourmet_Food_5.json.gz \
         --epochs 10 --learning-rate 0.001 --factors 5

mv results results6