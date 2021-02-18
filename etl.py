import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS','AWS_SECRET_ACCESS_KEY')

def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    song_data = 'data/song_data/*/*/*/*.json'
    print(song_data)
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table with columns song_id, title, artist_id, year, duration
    songs_table = df.select('song_id','title','artist_id','year','duration').dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist      
    songs_table.repartition("year").write.partitionBy("artist_id").parquet(output_data)


    # extract columns to create artists table with columns artist_id, name, location, lattitude, longitude
    artists_table = df.select('artist_id','artist_name','artist_location','artist_lattitude','artist_longitude')\
                    .withColumnRenamed('artist_name','name')\
                    .withColumnRenamed('artist_location','location')\
                    .withColumnRenamed('artist_lattitude','lattitude')\
                    .withColumnRenamed('artist_longitude','longitude')\
                    .dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data)


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = input_data + 'log_data'
    log_data = 'data/*.json'

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table with columns user_id, first_name, last_name, gender, level (with the most recent status)
    users_table = df.select('userId','firstName','lastName','gender','level')\
                    .withColumnRenamed('userID','user_id')\
                    .withColumnRenamed('firstName','first_name')\
                    .withColumnRenamed('lastName','last_name')\
                    .dropDuplicates()
    
    # write users table to parquet files
    users_table.write.parquet(output_data)


    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: to_date(x), TimestampType())
    df = df.withColumn('start_time', get_timestamp(df.ts))

    
    # extract columns to create time table with columns start_time, hour, day, week, month, year, weekday
    df = df.withColumn('hour',hour('timestamp'))\
           .withColumn('day',dayofmonth('timestamp'))\
           .withColumn('week',weekofyear('timestamp'))\
           .withColumn('month', month('timestamp'))\
           .withColumn('year',year('timestamp'))\
           .withColumn('weekday',dayofweek('timestamp'))\
           .dropDuplicates()
    
    time_table = df.select('start_time','hour','day','week','month','year','weekday').dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.repartition("year").write.partitionBy("month").parquet(output_data)

    # read in song data to use for songplays table with columns 
    song_data = input_data + 'song_data/*/*/*/*.json'
    df = spark.read.json(song_data)
    
    song_df = df.select('song_id','title','artist_id','year','duration').dropDuplicates()

    # extract columns from joined song and log datasets to create songplays table with columns songplay_id, start_time(time_table), user_id(users_table), level(users_table), song_id(songs_table), artist_id(songs_table), session_id(df.sessionId), location(df.location), user_agent(df.userAgent)
    
    songplays_table = None

    # write songplays table to parquet files partitioned by year and month
    songplays_table.repartition("year").write.partitionBy("month").parquet(output_data)


def main():
    print('.---------0.')
    spark = create_spark_session()
    print()
    print('  ...........1.')
    input_data = "s3://udacity-dend/"
    output_data = "output_data/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
