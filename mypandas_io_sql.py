import pandas.io.sql as pandas_io_sql
import psycopg2
import string

def mytable_exists(name, con, flavor):
    flavor_map = {
        'sqlite': ("SELECT name FROM sqlite_master "
                   "WHERE type='table' AND name='%s';") % name,
        'mysql' : "SHOW TABLES LIKE '%s'" % name,
        'postgresql': "SELECT table_name FROM information_schema.tables WHERE table_name = '%s'" % name
    }
    query = flavor_map.get(flavor, None)
    if query is None:
        raise NotImplementedError
    #print "query:%s".format(query)
    return len(pandas_io_sql.tquery(query, con)) > 0
        
def mywrite_postgresql(frame, table, names, cur):
    bracketed_names = [column for column in names]
    col_names = ','.join(bracketed_names)
    wildcards = ','.join([r'%s'] * len(names))
    insert_query = "INSERT INTO %s (%s) VALUES (%s)" % (
        table, col_names, wildcards)
    data = [tuple(x) for x in frame.values]
    cur.executemany(insert_query, data)
    
def mywrite_frame(frame, name, con, flavor='sqlite', if_exists='fail', commit=True, **kwargs):
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    frame: DataFrame
    name: name of SQL table
    conn: an open SQL database connection object
    flavor: {'sqlite', 'mysql', 'oracle'}, default 'sqlite'
    if_exists: {'fail', 'replace', 'append'}, default 'fail'
        fail: If table exists, do nothing.
        replace: If table exists, drop it, recreate it, and insert data.
        append: If table exists, insert data. Create if does not exist.
    """

    if 'append' in kwargs:
        import warnings
        warnings.warn("append is deprecated, use if_exists instead",
                      FutureWarning)
        if kwargs['append']:
            if_exists='append'
        else:
            if_exists='fail'
    exists = mytable_exists(name, con, flavor)
    if if_exists == 'fail' and exists:
        raise ValueError, "Table '%s' already exists." % name

    #create or drop-recreate if necessary
    create = None
    if exists and if_exists == 'replace':
        create = "DROP TABLE %s" % name
    elif not exists:
        create = get_schema(frame, name, flavor)

    if create is not None:
        cur = con.cursor()
        cur.execute(create)
        cur.close()

    cur = con.cursor()
    # Replace spaces in DataFrame column names with _.
    safe_names = [s.replace(' ', '_').strip() for s in frame.columns]
    flavor_picker = {'sqlite' : pandas_io_sql._write_sqlite,
                     'mysql' : pandas_io_sql._write_mysql,
                     'postgresql' : mywrite_postgresql
                     }

    func = flavor_picker.get(flavor, None)
    if func is None:
        raise NotImplementedError
    func(frame, name, safe_names, cur)
    cur.close()
    if commit:
        con.commit()

def myupdate_db_table_with_frame(db_tablename, pd_frame, pkeys, db_connection, commit=True):

    upd_sql = "UPDATE %s SET " % db_tablename

    col_dtypes = pd_frame.dtypes
    for row_ix, row in pd_frame.iterrows():
        row_upd_sql = upd_sql + " ,".join(["%s = %s " % (col, row[col])
                                            if col_dtypes[col_ix] != 'O'	# Object
                                            #else "%s = \"%s \"" % (col, row[col])
                                            else "%s = $$%s$$" % (col, row[col])
                                            for col_ix, col in enumerate(pd_frame.columns)
                                            ])
        #for col in pd_frame.columns:
        #	row_upd_sql += "%s = %s " % (col, row[col])
        if type(pkeys) == type('str'):
            row_upd_sql += " WHERE %s = %s"	% (pkeys, row_ix)
        else:
            row_upd_sql += " WHERE "
            row_upd_sql += " AND ".join(["%s = %s "  % (key, row_ix[key_ix])
                                         for key_ix, key in enumerate(pkeys)])

        #print "in myupdate_db_table_with_frame: upd_sql={0}".format(row_upd_sql)
        db_connection.cursor().execute(row_upd_sql)

    db_connection.cursor().close()
    if commit:
        db_connection.commit()

    return pd_frame.shape[0]

def myupdate_db_table_with_series(db_tablename, pd_series, pkeys, col_name, db_connection, commit=True):

    upd_sql = "UPDATE %s SET " % db_tablename

    col_dtypes = pd_series.dtypes
    for row_ix, row_val in pd_series.iteritems():
        if type(col_dtypes) != type('str'):		# not a string
            row_upd_sql = upd_sql + "%s = %s " % (col_name, row_val)
        else:
            row_upd_sql = upd_sql + "%s = $$%s$$ " % (col_name, row_val)

        if type(pkeys) == type('str'):
            row_upd_sql += " WHERE %s = %s"	% (pkeys, row_ix)
        else:
            row_upd_sql += " WHERE "
            row_upd_sql += " AND ".join(["%s = %s "  % (key, row_ix[key_ix])
                                         for key_ix, key in enumerate(pkeys)])

        #print "in myupdate_db_table_with_frame: upd_sql={0}".format(row_upd_sql)
        db_connection.cursor().execute(row_upd_sql)

    db_connection.cursor().close()
    if commit:
        db_connection.commit()

    return pd_series.shape[0]

def myalter_db(sql, db_connection):
    sql = " ".join(string.split(sql))
    try:
        db_connection.cursor().execute(sql)
    except psycopg2.ProgrammingError:
        #print "DBException ignored:{0}".format(sql)
        db_connection.rollback()
    else:
        print "DB Alteration completed:{0}".format(sql)
        db_connection.commit()

def _replace_sql_limit(orig_sql, rpl_lmt):
    #print "mypandas_io_sql._replace_sql_limit: rpl_limit={0:,}".format(rpl_lmt)
    limit_pos = orig_sql.find('LIMIT')
    if limit_pos < 0 or rpl_lmt == 0:
        return orig_sql
    else:
        return orig_sql.replace(orig_sql[limit_pos + len('LIMIT'):].split()[0]
                                ,str(rpl_lmt)) + ';'

def myread_frame(sql, db_connection, index_col=None, verbose=0, tm_start=0, rpl_lmt=0):
    import datetime as tm

    #print "mypandas_io_sql.myread_frame: rpl_limit={0:,}".format(rpl_lmt)
    sql = " ".join(string.split(sql))
    sql = _replace_sql_limit(sql, rpl_lmt)
    print "[{0}]		executing:{1}...".format(str(tm.datetime.now()), sql)
    obs_df = pandas_io_sql.read_frame(sql, db_connection, index_col)
    if obs_df.shape[0] == 0 or index_col is None:
        print "[{0}]		read {1:,} obs".format(str(tm.datetime.now() - tm_start) + ';'
                                                + str(tm.datetime.now()), obs_df.shape[0])
    else:
        print "[{0}] 		read {1:,} obs; index = {2}..{3}".format(str(tm.datetime.now() - tm_start) + ';'
                                                + str(tm.datetime.now()), obs_df.shape[0], obs_df.index[0]
                                                                ,obs_df.index[-1])
        if verbose >= 1:
            print "sample rows:\n{0}".format(obs_df[::obs_df.shape[0]/5])

    return obs_df
    

