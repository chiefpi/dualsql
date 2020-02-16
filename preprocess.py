# what do we preprocess?

import os
import json
import shutil


def read_db_split(data_dir):
    
    
    return train_db, dev_db


def preprocess(dataset, remove_from=False):
    # validate output_vocab TODO: why remove?
    output_vocab = ['_UNK', '_EOS', '.', 't1', 't2', '=', 'select', 'from', 'as', 'value', 'join', 'on', ')', '(', 'where', 't3', 'by', ',', 'count', 'group', 'order', 'distinct', 't4', 'and', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 't5', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', 't7', '+', '/']
    if remove_from:
        output_vocab = ['_UNK', '_EOS', '=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group_by', 'order_by', 'distinct', 'and', 'limit_value', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!=', 'union', 'between', '-', '+', '/']
    print('size of output_vocab', len(output_vocab))
    print('output_vocab', output_vocab)
    print()

    if dataset == 'spider':
        spider_dir = 'data/spider/'
        database_schema_filename = 'data/spider/tables.json'
        output_dir = 'data/spider_data'
        if remove_from:
            output_dir = 'data/spider_data_removefrom'
        train_database, dev_database = read_db_split(spider_dir)
    elif dataset == 'sparc':
        sparc_dir = 'data/sparc/'
        database_schema_filename = 'data/sparc/tables.json'
        output_dir = 'data/sparc_data'
        if remove_from:
            output_dir = 'data/sparc_data_removefrom'
        train_database, dev_database = read_db_split(sparc_dir)
    elif dataset == 'cosql':
        cosql_dir = 'data/cosql/'
        database_schema_filename = 'data/cosql/tables.json'
        output_dir = 'data/cosql_data'
        if remove_from:
            output_dir = 'data/cosql_data_removefrom'
        train_database, dev_database = read_db_split(cosql_dir)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    schema_tokens = {}
    column_names = {}
    database_schemas = {}

    print('Reading spider database schema file')
    schema_tokens, column_names, database_schemas = read_database_schema(database_schema_filename, schema_tokens, column_names, database_schemas)
    num_database = len(schema_tokens)
    print('num_database', num_database, len(train_database), len(dev_database))
    print('total number of schema_tokens / databases:', len(schema_tokens))

    output_database_schema_filename = os.path.join(output_dir, 'tables.json')
    with open(output_database_schema_filename, 'w') as outfile:
        json.dump([v for k,v in list(database_schemas.items())], outfile, indent=4)

    # TODO: what is interaction list
    if dataset == 'spider':
        interaction_list = read_spider(spider_dir, database_schemas, column_names, output_vocab, schema_tokens, remove_from)
    elif dataset == 'sparc':
        interaction_list = read_sparc(sparc_dir, database_schemas, column_names, output_vocab, schema_tokens, remove_from)
    elif dataset == 'cosql':
        interaction_list = read_cosql(cosql_dir, database_schemas, column_names, output_vocab, schema_tokens, remove_from)

    print('interaction_list length', len(interaction_list))

    train_interaction = []
    for database_id in interaction_list:
        if database_id not in dev_database:
            train_interaction += interaction_list[database_id]

    dev_interaction = []
    for database_id in dev_database:
        dev_interaction += interaction_list[database_id]

    print('train interaction: ', len(train_interaction))
    print('dev interaction: ', len(dev_interaction))

    write_interaction(train_interaction, 'train', output_dir)
    write_interaction(dev_interaction, 'dev', output_dir)

    return