#
#
#
mkdir results

python3 train.py --input data/Books_5.json.gz \
         --epochs 400 --learning-rate 0.1

python3 train.py --input data/Clothing_Shoes_and_Jewelry_5.json.gz \
         --epochs 250 --learning-rate 0.1

python3 train.py --input data/Home_and_Kitchen_5.json.gz \
         --epochs 200 --learning-rate 0.1

python3 train.py --input data/Electronics_5.json.gz \
         --epochs 200 --learning-rate 0.1

python3 train.py --input data/Movies_and_TV_5.json.gz \
         --epochs 100 --learning-rate 0.1

python3 train.py --input data/Kindle_Store_5.json.gz \
         --epochs 80 --learning-rate 0.1

python3 train.py --input data/Automotive_5.json.gz \
         --epochs 80 --learning-rate 0.1

python3 train.py --input data/Grocery_and_Gourmet_Food_5.json.gz \
         --epochs 80 --learning-rate 0.1

mv results results2
mkdir results

python3 train.py --input data/Books_5.json.gz \
         --epochs 400 --learning-rate 0.1 --factors 8

python3 train.py --input data/Clothing_Shoes_and_Jewelry_5.json.gz \
         --epochs 250 --learning-rate 0.1 --factors 8

python3 train.py --input data/Home_and_Kitchen_5.json.gz \
         --epochs 200 --learning-rate 0.1 --factors 8

python3 train.py --input data/Electronics_5.json.gz \
         --epochs 200 --learning-rate 0.1 --factors 8

python3 train.py --input data/Movies_and_TV_5.json.gz \
         --epochs 100 --learning-rate 0.1 --factors 8

python3 train.py --input data/Kindle_Store_5.json.gz \
         --epochs 80 --learning-rate 0.1 --factors 8

python3 train.py --input data/Automotive_5.json.gz \
         --epochs 80 --learning-rate 0.1 --factors 8

python3 train.py --input data/Grocery_and_Gourmet_Food_5.json.gz \
         --epochs 80 --learning-rate 0.1 --factors 8

mv results results3
mkdir results
