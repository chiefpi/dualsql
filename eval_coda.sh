pip install -r requirements.txt
python nltk_downloader.py
python eval.py --gold dev_gold.sql --pred predicted_sql.txt --etype match --db data/database --table data/tables.json