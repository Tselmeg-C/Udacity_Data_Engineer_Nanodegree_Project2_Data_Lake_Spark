import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, dayofweek,hour, weekofyear, date_format, monotonically_increasing_id
from pyspark.sql.types import TimestampType, DateType


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
    """
    Load song_data from S3 and create the songs_table and artists_table
    and save them back to S3.
    
    Params:
        spark        : Spark Session
        input_data   : S3 Location of song_data
        output_data  : S3 Location of tables to be saved. 
        
    """

    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
        
    # read song data file
    df = spark.read.json(song_data)
    df.createOrReplaceTempView('df_song')

    # extract columns to create songs table with columns song_id, title, artist_id, year, duration
    songs_table = df.select('song_id','title','artist_id','year','duration').dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist      
    songs_table.write.partitionBy('year','artist_id').parquet(output_data + 'songs/')

    # extract columns to create artists table with columns artist_id, name, location, lattitude, longitude
    artists_table = df.select('artist_id','artist_name','artist_location','artist_latitude','artist_longitude') \
                    .withColumnRenamed('artist_name','name') \
                    .withColumnRenamed('artist_location','location') \
                    .withColumnRenamed('artist_latitude','latitude') \
                    .withColumnRenamed('artist_longitude','longitude') \
                    .dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists/')


def process_log_data(spark, input_data, output_data):
    """
    Load log_data from public S3 bucket to create the users_table, time_table and songplays_table
    and save them back to S3.
    
    Params:
        spark        : Spark Session
        input_data   : S3 Location of song_data
        output_data  : S3 Location of tables to be saved. 
    """
    
    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'
    
    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table with columns user_id, first_name, last_name, gender, level 
    users_table = df.select('userId','firstName','lastName','gender','level')\
                    .withColumnRenamed('userID','user_id')\
                    .withColumnRenamed('firstName','first_name')\
                    .withColumnRenamed('lastName','last_name')\
                    .dropDuplicates()
    
    # write users table to parquet files
    users_table.write.parquet(output_data + 'users/')


    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: to_date(x), TimestampType())
    df = df.withColumn('start_time', get_timestamp(df.ts))

    
    # extract columns to create time table with columns start_time, hour, day, week, month, year, weekday
    df = df.withColumn('hour',hour('timestamp')) \
                 .withColumn('day',dayofmonth('timestamp')) \
                 .withColumn('week',weekofyear('timestamp')) \
                 .withColumn('month', month('timestamp')) \
                 .withColumn('year',year('timestamp')) \
                 .withColumn('weekday',dayofweek('timestamp')) \
                 .dropDuplicates()
    
    time_table = df.select('start_time','hour','day','week','month','year','weekday').dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year','month').parquet(output_data + 'time/')

    # read in song data to use for songplays table with columns   
    df_song = df_song.select('song_id','artist_id','artist_name','title').dropDuplicates()

    # extract columns from joined song and log datasets to create songplays table with columns songplay_id (serial number to be created), start_time(log_data), user_id(log_data), level(log_data), song_id(song_data), artist_id(song_data), session_id(log_data), location(log_data), user_agent(log_data)
    
    songplays_table = df.join(df_song, df.artist == df_song.artist_name, 'left') \
                    .distinct() \
                    .select(col('start_time'), col('userId'), col('level'), col('sessionId'), \
                                       col('year'),col('month'),col('location'), col('userAgent'), col('song_id'), col('artist_id'))\
                    .withColumn('songplay_id', monotonically_increasing_id()) \
                    .withColumnRenamed('userId','user_id') \
                    .withColumnRenamed('sessionId','session_id') \
                    .withColumnRenamed('userAgent','user_agent')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year','month').parquet(output_data + 'songplays/')


def main():
    spark = create_spark_session()
    input_data = "s3://udacity-dend/"
    output_data = "s3://udacitydlproject/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
